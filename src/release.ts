/**
 * Reference training pipeline: from a traffic sample to a paired, validated release.
 *
 *   plan   traffic + labelled history -> embed -> (teacher scores) -> a stratified probability sample
 *          (top band x2 when an independent teacher scores the frame) -> one blinded labelling queue.
 *          Writes <out>/plan/: queue.json (for reviewers: item id, text, heads), key.json (private:
 *          which record each item is) and plan.json (how the traffic was split).
 *   train  reviews -> records -> firing rules (certified on calibration negatives) and JOINT dismissal
 *          rules (certified on calibration positives) -> trainHeads (type 'auto', dismissal, int8
 *          references, guaranteed thresholds by default) -> buildArtifact -> round trip and validation.
 *          Writes <out>/release/: ruleset.json, artifact.json, manifest.json (pairs them: the rule
 *          set's hash, the artifact's hash, the embedding spec) and report.md (gates, guarantees, the
 *          share of traffic the rules tier settles).
 *
 * Embedding: the multi-layer reference embedder (src/embed-layers.ts) by default, cached on disk, so
 * training and serving compute the same features.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  applyReviews, assertRoundTrip, buildArtifact, checkEmbeddingSpec, designOf, designSample, labelQueue, reportMarkdown, settleWithRules,
  trainHeads, validateArtifact, type DecisionPolicy, type EmbeddingSpec, type ExampleRecord, type FrameItem, type HeadPolicy, type QueueKey, type Split,
} from '@liquidau/embedding-classifier';
import { CachedEmbedder } from '@liquidau/embedding-classifier/embedder';
import {
  buildRuleSet, certifyDismissals, certifyFalseAlarms, hitMatrix, jointLabels, mineDismissals, mineRules, ruleSetHash, ruleSetMatcher,
  validateDismissals, validateRules, type MinedRule, type RecordLike,
} from '@liquidau/rule-miner';
import { decisionFunction, fitLogistic, fitPlatt, predictPlatt, seededRandom, sigmoid } from '@liquidau/solvers';

import { makeLayerEmbedder } from './embed-layers.ts';

export interface Message { id: string; text: string; group?: string }
export interface LabelledMessage extends Message { labels: Record<string, 0 | 1> }
export interface HeadConfig {
  name: string;
  kind: 'recall' | 'precision';
  target: number;
  /** Mine firing rules for this head. */
  rules?: boolean;
  /**
   * The largest share of ordinary traffic the head may fire on (measured on the held-back budget
   * traffic, which holds positives at their natural rate, so set it above prevalence × recall). A
   * promise that needs a lower threshold fails the head's gates instead of shipping with unusable
   * false alarms.
   */
  maxFireRate?: number;
}
export interface ReleaseConfig {
  name: string;
  version: string;
  heads: HeadConfig[];
  policy: DecisionPolicy<string>;
  /** Dismissal rules: 'joint' (one "none of the heads" rule set, the default) or 'none'. */
  dismissal?: 'joint' | 'none';
  /** Labels to spend on the main sample (default 1800). */
  labels?: number;
  /** The multi-layer ONNX export (default .cache/onnx/mpnet-layers.onnx) and its layers. */
  embedding?: { model?: string; layers?: number[]; maxChars?: number };
  /**
   * How the labelling sample is stratified (with rule firing): 'primary' (default) - the first rule
   * head's score (else the first head's); 'max' - the highest score over heads; 'per-head' - the
   * primary score crossed with every other head's score band. 'primary' usually certifies the most
   * rules: the sample lands where the rule head's positives are, and its strata stay well filled.
   */
  stratify?: 'max' | 'primary' | 'per-head';
  /**
   * How candidate rules become certified rules (the certification always uses the calibration sample):
   * 'prefix' (default) - the mined order, the longest certified prefix (one weak early rule ends it);
   * 'select' - first keep only rules with no errors on the training data (the history plus the
   * sample's training split, never calibration), ordered by how much they cover there, then the
   * certified prefix; 'select-union' - the same selection, then the selected set's union certified
   * as ONE statement (all or nothing).
   */
  ruleSelection?: 'prefix' | 'select' | 'select-union';
  /** An independent offline teacher's probability per head for a text (e.g. Clef's cached answers). */
  teacher?: (text: string) => Record<string, number> | undefined;
  seed?: number;
  out: string;
  log?: (line: string) => void;
}

const ROOT = join(import.meta.dirname, '..');
const groupOf = (m: Message) => m.group ?? m.id;
const lengthBucket = (text: string) => (text.length < 60 ? 'short' : text.length < 200 ? 'medium' : 'long');

async function embedderFor(cfg: ReleaseConfig) {
  const layers = cfg.embedding?.layers ?? [4, 8, 12];
  const model = cfg.embedding?.model ?? join(ROOT, '.cache', 'onnx', 'mpnet-layers.onnx');
  const inner = await makeLayerEmbedder(model, { layers, maxChars: cfg.embedding?.maxChars ?? 2000 });
  const cache = new CachedEmbedder({
    cachePath: join(ROOT, '.cache', 'release', `emb-${layers.join('-')}`), embedBatch: (texts) => inner.embed(texts), batchSize: 16,
    maxChars: inner.spec.max_chars, dimensions: inner.spec.dimensions, format: 'binary', log: cfg.log,
  });
  return { spec: inner.spec, embedMany: (texts: string[]) => cache.embedMany(texts) };
}

/** Rules for one head: mined on 70% of the history, validated on the rest, vetoed by background traffic. */
function mineFiring(history: LabelledMessage[], head: string, veto: RecordLike[]): MinedRule[] {
  const mine = history.filter((_, i) => i % 10 < 7), validate = history.filter((_, i) => i % 10 >= 7);
  const mined = mineRules({ texts: mine.map((m) => m.text), y: mine.map((m) => m.labels[head] ?? null), groups: mine.map(groupOf), backgroundRecords: veto }, { minGroups: 3, maxBackgroundRate: 0.01, maxRules: 25 });
  return validateRules(mined, validate.map((m) => m.text), validate.map((m) => m.labels[head] ?? null), { maxFalsePositives: 1 }).kept;
}

/**
 * Rules with no errors on the selection data, most coverage first. kind 'fire': errors are negatives
 * hit, coverage positives hit; 'dismiss': errors are positives hit, coverage negatives hit.
 */
function selectRules(rules: MinedRule[], texts: readonly string[], y: ReadonlyArray<0 | 1 | null>, kind: 'fire' | 'dismiss'): MinedRule[] {
  const hits = hitMatrix(rules, texts);
  const bad = kind === 'fire' ? 0 : 1;
  return rules
    .map((r, k) => ({ r, errors: hits[k].filter((h, i) => h && y[i] === bad).length, cover: hits[k].filter((h, i) => h && y[i] === 1 - bad).length }))
    .filter((x) => x.errors === 0)
    .sort((a, b) => b.cover - a.cover)
    .map((x) => x.r);
}
const unionRow = (hits: readonly boolean[][]) => (hits.length ? hits[0].map((_, i) => hits.some((row) => row[i])) : []);

interface PlanFile { veto: string[]; budget: string[]; frame: string[]; stratumOf: Record<string, string>; design: ReturnType<typeof designSample>['design']; labels: number }

export async function plan(cfg: ReleaseConfig, traffic: Message[], history: LabelledMessage[]): Promise<{ queue: ReturnType<typeof labelQueue>['review'] }> {
  const log = cfg.log ?? (() => {});
  const seed = cfg.seed ?? 7;
  const rand = seededRandom(seed);
  // Split the traffic by group: veto and budget background (unlabelled), and the frame to sample from.
  const byGroup = new Map<string, Message[]>();
  for (const m of traffic) (byGroup.get(groupOf(m)) ?? byGroup.set(groupOf(m), []).get(groupOf(m))!).push(m);
  const groups = [...byGroup.values()];
  for (let i = groups.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [groups[i], groups[j]] = [groups[j], groups[i]]; }
  const vetoSet = groups.splice(0, 500).flat(), budgetSet = groups.splice(0, 500).flat(), frameSet = groups.flat();
  const veto: RecordLike[] = vetoSet.map((m) => ({ id: m.id, text: m.text, group: groupOf(m), role: 'background', backgroundUse: 'veto', source: { kind: 'traffic' }, labels: {} }));

  const embedder = await embedderFor(cfg);
  const vectors = await embedder.embedMany([...history, ...frameSet].map((m) => m.text));
  const emb = new Map([...history, ...frameSet].map((m, i) => [m.id, vectors[i]]));

  // Stratifying score: the teacher's (recalibrated on the history), else a round-0 head per label; max over labels.
  const heads = cfg.heads.map((h) => h.name);
  const logit = (p: number) => Math.log(Math.max(1e-6, p) / Math.max(1e-6, 1 - p));
  let perHead: (m: Message) => number[], scoringModel: string, topFactor = 1;
  if (cfg.teacher) {
    const platts = heads.map((h) => {
      const rows = history.filter((m) => cfg.teacher!(m.text)?.[h] !== undefined && (m.labels[h] === 0 || m.labels[h] === 1));
      return { h, platt: fitPlatt(rows.map((m) => logit(cfg.teacher!(m.text)![h])), rows.map((m) => m.labels[h])) };
    });
    perHead = (m) => platts.map(({ h, platt }) => { const v = cfg.teacher!(m.text)?.[h]; return v === undefined ? 0 : predictPlatt(platt, logit(v)); });
    scoringModel = 'teacher (recalibrated on the history)';
    topFactor = 2;
  } else {
    const models = heads.map((h) => { const rows = history.filter((m) => m.labels[h] === 0 || m.labels[h] === 1); return fitLogistic(rows.map((m) => emb.get(m.id)!), rows.map((m) => m.labels[h]), { C: 1, classWeight: 'balanced' }); });
    perHead = (m) => models.map((mod) => sigmoid(decisionFunction(mod, emb.get(m.id)!)));
    scoringModel = 'round-0 heads (history)';
  }
  const stratify = cfg.stratify ?? 'primary';
  const primary = Math.max(0, heads.indexOf(cfg.heads.find((h) => h.rules)?.name ?? heads[0]));
  const band = (p: number) => (p >= 0.9 ? 'high' : p >= 0.5 ? 'mid' : 'low');
  const score = (m: Message) => { const s = perHead(m); return stratify === 'max' ? Math.max(...s) : s[primary]; };
  const sliceOf = (m: Message) => (stratify === 'per-head' ? heads.map((h, i) => (i === primary ? '' : `${h}:${band(perHead(m)[i])}`)).filter(Boolean).join('|') || 'all' : lengthBucket(m.text));
  scoringModel += `, stratified ${stratify === 'max' ? 'by the highest head score' : stratify === 'primary' ? `by ${heads[primary]}` : `by ${heads[primary]} and every other head's band`}`;
  const candidates = cfg.heads.filter((h) => h.rules).flatMap((h) => mineFiring(history, h.name, veto));
  const fires = (text: string) => candidates.length > 0 && hitMatrix(candidates, [text]).some((row) => row[0]);
  const frame: FrameItem[] = frameSet.map((m) => ({ id: m.id, text: m.text, group: groupOf(m), signals: { ruleFires: fires(m.text), score: score(m), slice: sliceOf(m) } }));
  const labels = Math.min(frame.length, cfg.labels ?? 1800);
  // 'per-head' crosses the strata with the other heads' bands (signals.slice); the other options don't stratify by slice.
  const main = designSample({ frame, scoreBands: [0.9, 0.5], scoringModel, mergeSmallStrata: true, ...(stratify === 'per-head' ? { bySlice: true } : {}), roleSplit: { train: 0.2, calibration: 0.4, test: 0.4 }, allocation: { total: labels, method: 'proportional', topFactor }, seed: 202 + seed });
  for (const w of main.design.warnings) log(`design: ${w}`);
  const queue = labelQueue({ items: main.records.map((record) => ({ record, mechanism: 'sampled' as const, heads })), budget: { sampled: main.records.length, retrieved: 0, verification: 0 }, seed: 303 + seed });
  const dir = join(cfg.out, 'plan');
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'queue.json'), JSON.stringify(queue.review, null, 1));
  writeFileSync(join(dir, 'key.json'), JSON.stringify(queue.key));
  const planFile: PlanFile = { veto: vetoSet.map((m) => m.id), budget: budgetSet.map((m) => m.id), frame: frameSet.map((m) => m.id), stratumOf: main.stratumOf, design: main.design, labels: main.records.length };
  writeFileSync(join(dir, 'plan.json'), JSON.stringify(planFile));
  log(`plan: frame ${frame.length}, veto ${vetoSet.length}, budget ${budgetSet.length}; ${queue.review.length} items to label (${scoringModel}${topFactor !== 1 ? `, top band x${topFactor}` : ''})`);
  return { queue: queue.review };
}

export async function train(cfg: ReleaseConfig, traffic: Message[], history: LabelledMessage[], reviews: ReadonlyArray<{ itemId: string; labels: Record<string, 0 | 1> }>) {
  const log = cfg.log ?? (() => {});
  const dir = join(cfg.out, 'plan');
  const planFile = JSON.parse(readFileSync(join(dir, 'plan.json'), 'utf8')) as PlanFile;
  const key = JSON.parse(readFileSync(join(dir, 'key.json'), 'utf8')) as QueueKey;
  const byId = new Map(traffic.map((m) => [m.id, m]));
  const heads = cfg.heads.map((h) => h.name);
  const labelled = applyReviews(key, reviews);
  for (const w of labelled.warnings) log(`reviews: ${w}`);
  const records: ExampleRecord[] = [
    ...history.map((m): ExampleRecord => ({ id: m.id, text: m.text, group: groupOf(m), role: 'train', source: { kind: 'traffic' }, labels: { ...m.labels }, labelledBy: 'human' })),
    ...labelled.records,
  ];
  const veto: RecordLike[] = planFile.veto.map((id) => { const m = byId.get(id)!; return { id, text: m.text, group: groupOf(m), role: 'background', backgroundUse: 'veto', source: { kind: 'traffic' }, labels: {} }; });
  const budget: ExampleRecord[] = planFile.budget.map((id) => { const m = byId.get(id)!; return { id, text: m.text, group: groupOf(m), role: 'background', backgroundUse: 'budget', source: { kind: 'traffic' }, labels: {} }; });
  const cal = labelled.records.filter((r) => r.role === 'calibration');
  const framed = planFile.frame.filter((id) => planFile.stratumOf[id] !== undefined).map((id) => byId.get(id)!);

  // Firing rules per rule head, certified on the calibration negatives (and capped by their firing on the frame).
  const selection = cfg.ruleSelection ?? 'prefix';
  // Selection data: the history and the sample's training split - never calibration or test.
  const selectionRows = [...history.map((m) => ({ text: m.text, labels: m.labels as Record<string, 0 | 1 | null> })), ...labelled.records.filter((r) => r.role === 'train').map((r) => ({ text: r.text, labels: r.labels }))];
  const firing = cfg.heads.filter((h) => h.rules).map((h) => {
    const mined = mineFiring(history, h.name, veto);
    const kept = selection === 'prefix' ? mined : selectRules(mined, selectionRows.map((r) => r.text), selectionRows.map((r) => r.labels[h.name] ?? null), 'fire');
    const calH = cal.filter((r) => r.labels[h.name] === 0 || r.labels[h.name] === 1);
    const rows = (rules: MinedRule[]) => selection === 'select-union' ? [unionRow(hitMatrix(rules, calH.map((r) => r.text)))] : hitMatrix(rules, calH.map((r) => r.text));
    const frameRows = (rules: MinedRule[]) => selection === 'select-union' ? [unionRow(hitMatrix(rules, framed.map((m) => m.text)))] : hitMatrix(rules, framed.map((m) => m.text));
    const certified = kept.length ? certifyFalseAlarms(rows(kept), {
      y: calH.map((r) => r.labels[h.name] as 0 | 1), maxRate: 0.01, design: designOf(calH),
      frame: { strata: framed.map((m) => planFile.stratumOf[m.id]), fired: frameRows(kept) },
    }).certified : 0;
    const out = selection === 'select-union' ? (certified ? kept : []) : kept.slice(0, certified);
    log(`rules: ${h.name}: ${out.length} of ${mined.length} firing rules certified (${selection}${selection !== 'prefix' ? `, ${kept.length} selected` : ''})`);
    return { label: h.name, rules: out };
  });
  // Joint dismissal rules ("none of the heads"), certified on the calibration positives of any head.
  let dismiss: MinedRule[] = [];
  if ((cfg.dismissal ?? 'joint') === 'joint') {
    const mine = history.filter((_, i) => i % 10 < 7), validate = history.filter((_, i) => i % 10 >= 7);
    const mined = mineDismissals({ texts: mine.map((m) => m.text), y: jointLabels(mine.map((m) => m.labels), heads), groups: mine.map(groupOf) });
    const validated = validateDismissals(mined, validate.map((m) => m.text), jointLabels(validate.map((m) => m.labels), heads)).kept;
    const kept = selection === 'prefix' ? validated : selectRules(validated, selectionRows.map((r) => r.text), jointLabels(selectionRows.map((r) => r.labels), heads), 'dismiss');
    const yCal = jointLabels(cal.map((r) => r.labels), heads);
    const calJ = cal.filter((_, i) => yCal[i] !== null);
    const hitsJ = (rules: MinedRule[]) => (selection === 'select-union' ? [unionRow(hitMatrix(rules, calJ.map((r) => r.text)))] : hitMatrix(rules, calJ.map((r) => r.text)));
    const cert = kept.length && yCal.includes(1) ? certifyDismissals(hitsJ(kept), { y: jointLabels(calJ.map((r) => r.labels), heads), maxRate: 0.05, design: designOf(calJ) }).certified : 0;
    dismiss = selection === 'select-union' ? (cert ? kept : []) : kept.slice(0, cert);
    log(`rules: joint dismissal: ${dismiss.length} of ${validated.length} rules certified (${selection}${selection !== 'prefix' ? `, ${kept.length} selected` : ''})`);
  }
  const ruleSet = buildRuleSet(cfg.version, [...firing, ...(dismiss.length ? heads.map((label) => ({ label, effect: 'dismiss' as const, rules: dismiss })) : [])]);
  const ruleSetId = ruleSetHash(ruleSet);
  const matcher = ruleSetMatcher(ruleSet);
  const cleared = (text: string) => matcher.evaluate(text).dismissed.length === heads.length;

  // Heads: type 'auto', the guarantee each policy's data supports, dismissal counted as misses.
  const embedder = await embedderFor(cfg);
  const X = await embedder.embedMany(records.map((r) => r.text));
  // Firing caps, measured on the budget traffic (held back from mining and sampling).
  const caps = Object.fromEntries(cfg.heads.filter((h) => h.maxFireRate !== undefined).map((h) => [h.name, h.maxFireRate!]));
  const background = Object.keys(caps).length ? { X: await embedder.embedMany(budget.map((r) => r.text)), records: budget, maxRate: caps } : undefined;
  const policyOf = (h: HeadConfig): HeadPolicy => (h.kind === 'recall' ? { kind: 'recall', targetRecall: h.target, designMethod: 'linearised' } : { kind: 'precision', targetPrecision: h.target });
  const result = trainHeads({
    X, split: records.map((r) => r.role as Split), records, designs: [planFile.design], groups: records.map((r) => r.group), foldKeys: records.map((r) => r.group),
    referenceEncoding: 'int8', provenance: {}, ...(background ? { background } : {}),
    heads: cfg.heads.map((h) => ({
      name: h.name, y: records.map((r) => r.labels[h.name] ?? null), policy: policyOf(h), type: 'auto' as const,
      ...(dismiss.length ? { dismissal: { rows: records.map((r) => cleared(r.text)), background: budget.map((r) => cleared(r.text)), ruleSet: ruleSetId, maxRate: 0.05, certified: dismiss.length } } : {}),
    })),
    log,
  });
  const artifact = buildArtifact(result, { version: cfg.version, embedding: embedder.spec, training: { rule_set: ruleSetId, labels: planFile.labels } });
  assertRoundTrip(artifact, X, result.testProbabilities);
  validateArtifact(JSON.parse(JSON.stringify(artifact)));
  const enforceable = (() => { try { validateArtifact(artifact, { mode: 'enforce' }); return true; } catch { return false; } })();
  if (checkEmbeddingSpec(artifact, embedder.spec).length) throw new Error('embedding spec mismatch inside the release');

  // The rules tier's share of the frame (deterministic, no labels needed).
  const settled = planFile.frame.filter((id) => settleWithRules(cfg.policy, matcher.evaluate(byId.get(id)!.text)).settled).length / planFile.frame.length;
  const out = join(cfg.out, 'release');
  mkdirSync(out, { recursive: true });
  const artifactJson = JSON.stringify(artifact);
  const sha256 = (s: string) => createHash('sha256').update(s).digest('hex');
  writeFileSync(join(out, 'artifact.json'), artifactJson);
  writeFileSync(join(out, 'ruleset.json'), JSON.stringify(ruleSet, null, 1));
  const manifest = {
    name: cfg.name, version: cfg.version, created_at: artifact.created_at,
    rule_set: { file: 'ruleset.json', hash: ruleSetId, format: ruleSet.format, firing: firing.reduce((a, f) => a + f.rules.length, 0), dismissal: dismiss.length },
    artifact: { file: 'artifact.json', sha256: sha256(artifactJson), gates_passed: artifact.gates!.passed, enforceable },
    embedding: embedder.spec, policy: cfg.policy, labels_spent: planFile.labels,
    heads: Object.fromEntries(heads.map((h) => [h, { type: artifact.heads[h]!.features?.kind ?? 'linear', guarantee: artifact.heads[h]!.guarantee ?? null }])),
    rules_tier_share_of_frame: settled,
  };
  writeFileSync(join(out, 'manifest.json'), JSON.stringify(manifest, null, 1));
  const report = [reportMarkdown(artifact, { title: `${cfg.name} ${cfg.version}` }), '', '## Rules', '',
    `Rule set ${ruleSetId.slice(0, 12)} (${ruleSet.format}): ${manifest.rule_set.firing} firing rule(s) (certified: at most 1% of negatives, 95%), ${dismiss.length} joint dismissal rule(s) (at most 5% of positives, 95%).`,
    `The rules tier settles ${(100 * settled).toFixed(1)}% of the frame without the model (every policy head decided).`];
  writeFileSync(join(out, 'report.md'), report.join('\n'));
  log(`release: gates ${artifact.gates!.passed ? 'passed' : `FAILED (${artifact.gates!.failures.length})`}; rules tier ${(100 * settled).toFixed(1)}% of the frame; written to ${out}`);
  return { artifact, ruleSet, manifest, result };
}

/** Read a release back as a runtime would: validated, paired and with the expected embedder. */
export function loadRelease(dir: string, mode: 'shadow' | 'enforce', runtimeEmbedding: EmbeddingSpec) {
  const manifest = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8'));
  const artifact = validateArtifact(JSON.parse(readFileSync(join(dir, manifest.artifact.file), 'utf8')), { mode });
  const ruleSet = JSON.parse(readFileSync(join(dir, manifest.rule_set.file), 'utf8'));
  const problems = [...checkEmbeddingSpec(artifact, runtimeEmbedding), ...(ruleSetHash(ruleSet) !== manifest.rule_set.hash ? ['rule set does not match the manifest'] : [])];
  if (problems.length) throw new Error(`refusing to load release: ${problems.join('; ')}`);
  return { manifest, artifact, ruleSet };
}
export const releaseExists = (dir: string) => existsSync(join(dir, 'manifest.json'));
