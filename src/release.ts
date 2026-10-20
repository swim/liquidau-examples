/**
 * Reference training pipeline: from a traffic sample to a paired, validated release.
 *
 *   plan   traffic + labelled history -> embed -> (teacher scores) -> a stratified probability sample
 *          (top band x2 when an independent teacher scores the frame) -> one blinded labelling queue.
 *          Writes <out>/plan/: queue.json (for reviewers: item id, text, heads), key.json (private:
 *          which record each item is) and plan.json (how the traffic was split).
 *   train  reviews -> records -> firing rules (certified on calibration negatives) and JOINT dismissal
 *          rules (certified on calibration positives) -> trainHeads (type 'auto', dismissal, int8
 *          references, guaranteed thresholds by default) -> buildArtifact -> round trip and validation
 *          -> the final routing and per-head evaluation on the held-out test split -> a complete
 *          @liquidau/router release (buildRelease). Writes <out>/release/: manifest.json, classifier.json,
 *          rules.json, policy.json, evidence.json, deploy.json (the manifest digest a deployment pins),
 *          summary.json and report.md (gates, guarantees, evaluation, the rules tier's share).
 *   load   loadRelease serves the release through loadRouter, which validates and pairs everything.
 *
 * Embedding: the multi-layer reference embedder (src/embed-layers.ts) by default, cached on disk, so
 * training and serving compute the same features.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  applyReviews, assertRoundTrip, buildArtifact, checkEmbeddingSpec, designOf, designSample, labelQueue, reportMarkdown, settleWithRules,
  assembleEdge, edgeLadder, trainEdge, trainHeads, validateArtifact, type DecisionPolicy, type ExampleRecord, type FrameItem, type HeadPolicy, type QueueKey, type Split,
} from '@liquidau/embedding-classifier';
import {
  buildRuleSet, certifyDismissals, certifyFalseAlarms, hitMatrix, jointLabels, mineDismissals, mineRules, ruleSetHash, ruleSetMatcher,
  validateDismissals, validateRules, type MinedRule, type RecordLike,
} from '@liquidau/rule-miner';
import { createEdgeTier, loadRouter, type Encoder, type Router, type RouterEvent, type RouterPolicy } from '@liquidau/router';
import { decisionFunction, fitLogistic, fitPlatt, predictPlatt, seededRandom, sigmoid } from '@liquidau/solvers';

import { releaseEmbedder, type ReleaseEncoderOptions } from './release-encoder.ts';
import { fileSource, packageVersions, writeRelease } from './router-host.ts';
import { buildRouterRelease, policyFor } from './router-release.ts';
import { GATES } from './gates.ts';
import { staticEncoder } from './static-host.ts';

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
  /** Recall heads: recall held in reserve for later periods (HeadPolicy.reserve; default 0). */
  reserve?: number;
}
export interface ReleaseConfig {
  name: string;
  version: string;
  heads: HeadConfig[];
  policy: DecisionPolicy<string>;
  /** Dismissal rules: 'joint' (one "none of the heads" rule set, the default) or 'none'. */
  dismissal?: 'joint' | 'none';
  /**
   * The edge tier (embedding-classifier's trainEdge; on by default): regex + lexical + static scorers,
   * blended, that fire (≤ maxFalseAlarms of negatives, default 0.005) and dismiss (recall heads: ≤
   * recallShare × the miss budget, default 0.5; precision heads: ≤ precisionDismiss of positives,
   * default 0.075) before the embedding model; heads are certified jointly with it. `false` turns it off.
   */
  edge?: false | { maxFalseAlarms?: number; recallShare?: number; precisionDismiss?: number };
  /** Labels to spend on the main sample (default 1800). */
  labels?: number;
  /**
   * The main sample's train / calibration / test shares (default 0.2 / 0.4 / 0.4). With the finer-label
   * score, calibration 0.6 (0.1 / 0.6 / 0.3) helped every head in the lab's harness (stage SD), but on the
   * Banking77 walkthrough (mpnet, FINER=1) it failed both heads' calibration gates: check the gates
   * before relying on it.
   */
  roleSplit?: { train: number; calibration: number; test: number };
  /**
   * Extra labels spent on calibration only, as a share of `labels` (e.g. 0.5 for +50%); train and test
   * keep their sizes. The lab's best single use of a larger budget (stage SD: +50% to calibration helped
   * 3 heads and hurt none on EmbeddingGemma). Default 0.
   */
  extraCalibrationLabels?: number;
  /**
   * The encoder (src/release-encoder.ts): 'gemma' (default, EmbeddingGemma) or 'mpnet-layers' (the
   * multi-layer mpnet ONNX export, with `model` and `layers`).
   */
  embedding?: Omit<ReleaseEncoderOptions, 'log'>;
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

const groupOf = (m: Message) => m.group ?? m.id;
const lengthBucket = (text: string) => (text.length < 60 ? 'short' : text.length < 200 ? 'medium' : 'long');

const embedderFor = (cfg: ReleaseConfig) => releaseEmbedder({ ...cfg.embedding, log: cfg.log });

/** The router policy for a release: one route per head, the config's priority and suppression. */
export const routerPolicy = (cfg: Pick<ReleaseConfig, 'heads' | 'policy'>): RouterPolicy => policyFor(cfg.heads.map((h) => h.name), cfg.policy.priority, cfg.policy.suppress ?? []);

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
  const base = cfg.roleSplit ?? { train: 0.2, calibration: 0.4, test: 0.4 }, extra = cfg.extraCalibrationLabels ?? 0;
  if (![base.train, base.calibration, base.test].every((v) => v > 0) || Math.abs(base.train + base.calibration + base.test - 1) > 1e-9) throw new Error('roleSplit: train, calibration and test must be positive shares summing to 1');
  if (!(extra >= 0)) throw new Error('extraCalibrationLabels must be a non-negative share of labels');
  const labels = Math.min(frame.length, Math.round((cfg.labels ?? 1800) * (1 + extra)));
  // The extra labels all go to calibration: train and test keep the sizes the base budget gives them.
  const roleSplit = { train: base.train / (1 + extra), calibration: (base.calibration + extra) / (1 + extra), test: base.test / (1 + extra) };
  if (extra || cfg.roleSplit) log(`design: ${labels} labels, split ${roleSplit.train.toFixed(3)} / ${roleSplit.calibration.toFixed(3)} / ${roleSplit.test.toFixed(3)}`);
  // 'per-head' crosses the strata with the other heads' bands (signals.slice); the other options don't stratify by slice.
  const main = designSample({ frame, scoreBands: [0.9, 0.5], scoringModel, mergeSmallStrata: true, ...(stratify === 'per-head' ? { bySlice: true } : {}), roleSplit, allocation: { total: labels, method: 'proportional', topFactor }, seed: 202 + seed });
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
  const policyOf = (h: HeadConfig): HeadPolicy => (h.kind === 'recall' ? { kind: 'recall', targetRecall: h.target, designMethod: 'linearised', ...(h.reserve ? { reserve: h.reserve } : {}) } : { kind: 'precision', targetPrecision: h.target });
  // The edge tier: trained on the training split, certified on the calibration sample's design.
  const edgeCfg = cfg.edge === false ? null : { maxFalseAlarms: 0.005, recallShare: 0.5, precisionDismiss: 0.075, ...cfg.edge };
  const st = edgeCfg ? await staticEncoder() : null;
  const firingByLabel = new Map(firing.map((f) => [f.label, f.rules]));
  const ruleColumn = (label: string, texts: readonly string[]) => { const rules = firingByLabel.get(label) ?? []; if (!rules.length) return undefined; const hits = hitMatrix(rules, texts, ruleSet.lexicon); return texts.map((_, i) => hits.some((row) => row[i])); };
  const edge = edgeCfg ? trainEdge({
    texts: records.map((r) => r.text), split: records.map((r) => r.role as Split), groups: records.map((r) => r.group), design: designOf(cal), seed: cfg.seed ?? 0,
    static: { identity: st!.identity, vectors: records.map((r) => st!.embed(r.text)) },
    ...(background ? { background: { texts: budget.map((r) => r.text), static: budget.map((r) => st!.embed(r.text)) } } : {}),
    heads: cfg.heads.map((h) => {
      const col = ruleColumn(h.name, records.map((r) => r.text));
      return {
        name: h.name, y: records.map((r) => r.labels[h.name] ?? null), maxFalseAlarms: edgeCfg.maxFalseAlarms,
        // A ladder (full, half, fire-only): trainHeads keeps the first with which the head passes its gates.
        maxDismissed: [1, 0.5, 0].map((k) => k * (h.kind === 'recall' ? edgeCfg.recallShare * (1 - h.target) : edgeCfg.precisionDismiss)),
        ...(col ? { ruleFired: col, ...(background ? { backgroundRuleFired: ruleColumn(h.name, budget.map((r) => r.text)) } : {}) } : {}),
      };
    }),
  }) : null;
  for (const w of edge?.warnings ?? []) log(`edge: ${w}`);
  // Research hook (lab stage Q): RELEASE_EDGE_FIRST=<head,...> gives those heads the ladder's first candidate only.
  const edgeFirst = new Set((process.env.RELEASE_EDGE_FIRST ?? '').split(',').filter(Boolean));
  const result = trainHeads({
    X, split: records.map((r) => r.role as Split), records, designs: [planFile.design], groups: records.map((r) => r.group), foldKeys: records.map((r) => r.group),
    referenceEncoding: 'int8', provenance: {}, calibrationAlpha: GATES.calibrationAlpha, ...(background ? { background } : {}),
    heads: cfg.heads.map((h) => ({
      name: h.name, y: records.map((r) => r.labels[h.name] ?? null), policy: policyOf(h), type: 'auto' as const,
      ...(dismiss.length ? { dismissal: { rows: records.map((r) => cleared(r.text)), background: budget.map((r) => cleared(r.text)), ruleSet: ruleSetId, maxRate: 0.05, certified: dismiss.length } } : {}),
      ...(edge?.rungs[h.name] ? { edge: edgeLadder(edge, h.name).slice(0, edgeFirst.has(h.name) ? 1 : undefined).map((r) => ({ fired: r.fired, dismissed: r.dismissed, ...(background && r.background ? { background: r.background } : {}), edgeSha256: r.edgeSha256 })) } : {}),
    })),
    log,
  });
  // The edge document with each head's chosen candidate.
  const edgeDoc = edge ? assembleEdge(edge, result.edgeChoice ?? {}) : null;
  for (const [h, k] of Object.entries(result.edgeChoice ?? {})) if (k) log(`edge: ${h}: dismissal level ${k} kept (the head failed its gates with more dismissal)`);
  const artifact = buildArtifact(result, { version: cfg.version, embedding: embedder.spec, training: { rule_set: ruleSetId, labels: planFile.labels }, router: { ruleSetHash: ruleSetId } });
  assertRoundTrip(artifact, X, result.testProbabilities);
  validateArtifact(JSON.parse(JSON.stringify(artifact)));
  if (checkEmbeddingSpec(artifact, embedder.spec).length) throw new Error('embedding spec mismatch inside the release');

  // A complete router release, with evidence from the held-out test split (sampled, labelled, never used
  // to train or calibrate). Each head's own promise is its acceptance criterion.
  const encoder = await embedder.encoder();
  const { built, evidence, final } = await buildRouterRelease({
    releaseId: `${cfg.name}-${cfg.version}`, createdAt: artifact.created_at, artifact, ruleSet, policy: routerPolicy(cfg),
    items: records.map((r, i) => ({ id: r.id, text: r.text, x: X[i], labels: r.labels })).filter((_, i) => records[i].role === 'test'),
    dataset: { id: `${cfg.version}-test`, split: 'test', description: 'the labelled sample\'s held-out test split (a stratified probability sample; rates are unweighted point estimates)' },
    criteria: { heads: Object.fromEntries(cfg.heads.map((h) => [h.name, h.kind === 'recall' ? { minRecall: h.target } : { minPrecision: h.target }])), compare: 'point' },
    embedding: encoder.identity, packages: packageVersions(),
    ...(edgeDoc ? { edge: { document: edgeDoc, staticEncoder: st! } } : {}),
  });

  // The edge tier's share of the frame (rules and edge; deterministic, no labels needed).
  const tier = createEdgeTier(ruleSet, edgeDoc, routerPolicy(cfg).priority, st ?? undefined);
  const settled = planFile.frame.filter((id) => settleWithRules(cfg.policy, tier.evaluate(byId.get(id)!.text).findings).settled).length / planFile.frame.length;
  const out = join(cfg.out, 'release');
  writeRelease(out, built);
  const summary = {
    name: cfg.name, version: cfg.version, releaseId: built.manifest.releaseId, manifestSha256: built.manifestSha256,
    rule_set: { hash: ruleSetId, firing: firing.reduce((a, f) => a + f.rules.length, 0), dismissal: dismiss.length },
    gates_passed: built.manifest.gates.passed, enforcement: built.enforcement, labels_spent: planFile.labels,
    heads: Object.fromEntries(heads.map((h) => [h, { type: artifact.heads[h]!.features?.kind ?? 'linear', guarantee: artifact.heads[h]!.guarantee ?? null }])),
    rules_tier_share_of_frame: settled,
    edge: edgeDoc ? { heads: Object.fromEntries(Object.entries(edgeDoc.heads).map(([h, m]) => [h, { fire: m!.fire, dismiss: m!.dismiss, max_dismissed: m!.claims.max_dismissed, level: result.edgeChoice?.[h as string] ?? 0 }])) } : null,
  };
  writeFileSync(join(out, 'summary.json'), JSON.stringify(summary, null, 1));
  const pctOf = (r: { rate: number | null; count: number; of: number }) => (r.rate === null ? 'n/a' : `${(100 * r.rate).toFixed(1)}% (${r.count}/${r.of})`);
  const report = [reportMarkdown(artifact, { title: `${cfg.name} ${cfg.version}` }), '', '## Rules', '',
    `Rule set ${ruleSetId.slice(0, 12)} (${ruleSet.format}): ${summary.rule_set.firing} firing rule(s) (certified: at most 1% of negatives, 95%), ${dismiss.length} joint dismissal rule(s) (at most 5% of positives, 95%).`,
    `The rules tier settles ${(100 * settled).toFixed(1)}% of the frame without the model (every policy head decided).`,
    '', '## Release', '',
    `Router release ${built.manifest.releaseId}, manifest SHA-256 ${built.manifestSha256}.`,
    `Final routing on the test split (${final.items} items): coverage ${pctOf(final.coverage)}, routed accuracy ${pctOf(final.routedAccuracy)}, rules-path mismatches ${final.pathMismatches.length}.`,
    `Acceptance (each head's promise, point estimates on the test split): ${evidence.gates.passed ? 'passed' : `failed - ${evidence.gates.failures.join('; ')}`}.`,
    built.enforcement.length ? `Shadow only: ${built.enforcement.join('; ')}.` : 'Enforceable.'];
  writeFileSync(join(out, 'report.md'), report.join('\n'));
  log(`release: gates ${artifact.gates!.passed ? 'passed' : `FAILED (${artifact.gates!.failures.length})`}; ${built.enforcement.length ? 'shadow only' : 'enforceable'}; rules tier ${(100 * settled).toFixed(1)}% of the frame; written to ${out}`);
  return { artifact, ruleSet, built, summary, result };
}

/**
 * Serve a release as a runtime would: loadRouter validates every document, the pairing and the
 * encoder's identity, and refuses to enforce a release whose gates or evidence don't support it.
 */
export async function loadRelease(dir: string, options: { mode: 'shadow' | 'enforce'; encoder: Encoder; monitoringRate?: number; timeoutMs?: number; observer?: (event: RouterEvent) => void }): Promise<Router> {
  const deploy = JSON.parse(readFileSync(join(dir, 'deploy.json'), 'utf8')) as { manifestKey: string; manifestSha256: string };
  return loadRouter({
    staticEncoder,
    source: fileSource(dir), manifestKey: deploy.manifestKey, expectedManifestSha256: deploy.manifestSha256, mode: options.mode, encoder: options.encoder,
    timeoutMs: options.timeoutMs ?? 60_000, monitoring: { sampleRate: options.monitoringRate ?? 0, samplingSalt: 'liquidau-examples' },
    ...(options.observer ? { observer: options.observer } : {}),
  });
}
export const releaseExists = (dir: string) => existsSync(join(dir, 'deploy.json'));
