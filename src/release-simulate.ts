/**
 * The release pipeline end to end on a benchmark, as an acceptance test: plan -> a simulated
 * reviewer answers the queue from the true labels (skipping 2%) -> train (a complete router release)
 * -> serve it with @liquidau/router (loadRouter: enforce when the release supports it, else shadow)
 * -> route every message of the fully labelled future file, then check each head's promise, the final
 * routing and the router's decisions against the reference evaluator.
 *
 * Usage: release-simulate.ts <abcd|ade|toxicchat> [seed]   Writes out/release-sim/<dataset>/.
 * Env: RELEASE_ENCODER=gemma|mpnet-layers, RELEASE_SIM_TAG=<tag> (writes out/release-sim/<dataset>-<tag>/).
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { readFileSync } from 'node:fs';

import type { ClassifierArtifact } from '@liquidau/embedding-classifier';
import { createDecisionEvaluator, type RouteResult, type RouterEvent } from '@liquidau/router';
import type { RuleSet } from '@liquidau/rule-miner';
import { seededRandom } from '@liquidau/solvers';

import { loadAbcd, loadAde, type Doc } from './benchmarks.ts';
import { loadRelease, plan, routerPolicy, train, type LabelledMessage, type Message, type ReleaseConfig } from './release.ts';
import { ClefTeacher } from './teacher/clef.ts';
import { TOXICCHAT_QUESTIONS } from './teacher/questions.ts';
import { releaseEmbedderFor } from './release-encoder.ts';
import { loadToxicChat } from './toxicchat.ts';

const [name, seedArg] = process.argv.slice(2);
const seed = Number(seedArg ?? 7);
const ROOT = join(import.meta.dirname, '..');
const say = (line: string) => console.error(`[release ${name}] ${line}`);

let trainDocs: Doc[], future: Doc[], cfg: Omit<ReleaseConfig, 'out' | 'version'>;
if (name === 'abcd' || name === 'ade') {
  const b = name === 'abcd' ? loadAbcd() : loadAde();
  trainDocs = b.train; future = b.future;
  cfg = name === 'abcd'
    ? { name: 'abcd-support', heads: [{ name: 'access', kind: 'recall', target: 0.85, rules: true, maxFireRate: 0.2 }, { name: 'refund', kind: 'precision', target: 0.8, maxFireRate: 0.12 }], policy: { priority: ['access', 'refund'] } }
    : { name: 'ade-safety', heads: [{ name: 'ade', kind: 'recall', target: 0.85, rules: true, maxFireRate: 0.45 }], policy: { priority: ['ade'] }, dismissal: 'none' };
} else if (name === 'toxicchat') {
  const prompts = await loadToxicChat();
  const toDoc = (p: (typeof prompts)[number]): Doc => ({ id: p.id, text: p.text, group: p.group, truth: { ...p.truth } });
  trainDocs = prompts.filter((p) => p.split === 'train').map(toDoc);
  const seen = new Set(trainDocs.map((d) => d.group));
  future = prompts.filter((p) => p.split === 'test' && !seen.has(p.group)).map(toDoc);
  const clef = new ClefTeacher({ model: 'clef', questions: TOXICCHAT_QUESTIONS, cachePath: join(ROOT, 'teacher-labels', 'clef-toxicchat-clef.json') });
  cfg = { name: 'toxicchat-moderation', heads: [{ name: 'toxic', kind: 'recall', target: 0.8, rules: true, maxFireRate: 0.2 }, { name: 'jailbreak', kind: 'recall', target: 0.7, maxFireRate: 0.08 }], policy: { priority: ['jailbreak', 'toxic'] }, teacher: (text) => clef.cached(text) };
} else throw new Error('usage: release-simulate.ts <abcd|ade|toxicchat> [seed]');

// History: 1,500 labelled messages (whole groups); traffic: the rest of the train file, unlabelled.
const rand = seededRandom(seed * 13 + 1);
const byGroup = new Map<string, Doc[]>();
for (const d of trainDocs) (byGroup.get(d.group) ?? byGroup.set(d.group, []).get(d.group)!).push(d);
const order = [...byGroup.values()];
for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
const historyDocs: Doc[] = [];
while (historyDocs.length < 1500 && order.length) historyDocs.push(...order.shift()!);
const history: LabelledMessage[] = historyDocs.map((d) => ({ id: d.id, text: d.text, group: d.group, labels: { ...d.truth } }));
const traffic: Message[] = order.flat().map((d) => ({ id: d.id, text: d.text, group: d.group }));
// Research runs (several seeds or encoders side by side): RELEASE_ENCODER picks the release encoder
// ('gemma', the default, or 'mpnet-layers'); RELEASE_SIM_TAG writes to out/release-sim/<dataset>-<tag>/.
const encoder = process.env.RELEASE_ENCODER as 'gemma' | 'mpnet-layers' | undefined;
if (encoder !== undefined && encoder !== 'gemma' && encoder !== 'mpnet-layers') throw new Error(`RELEASE_ENCODER must be gemma or mpnet-layers, got ${encoder}`);
const out = join(ROOT, 'out', 'release-sim', process.env.RELEASE_SIM_TAG ? `${name}-${process.env.RELEASE_SIM_TAG}` : name);
const config: ReleaseConfig = { ...cfg, version: `${name}-${seed}`, seed, out, log: say, ...(encoder ? { embedding: { encoder } } : {}) };

const { queue } = await plan(config, traffic, history);
// The simulated reviewer reads each item's text (item ids are opaque) and answers from the true labels,
// skipping 2% of items as real reviewers do.
const byText = new Map(trainDocs.map((d) => [d.text, d.truth]));
const skipper = seededRandom(404 + seed);
const finalReviews = queue.filter(() => skipper() >= 0.02).map((item) => ({ itemId: item.itemId, labels: Object.fromEntries(item.heads.map((h) => [h, byText.get(item.text)![h]])) as Record<string, 0 | 1> }));
// Acceptance: serve the release with the router, as a runtime would, and route the whole future file.
const { built, summary } = await train(config, traffic, history, finalReviews);
const dir = join(out, 'release');
const heads = cfg.heads.map((h) => h.name);
const artifact = JSON.parse(readFileSync(join(dir, built.manifest.files.classifier.key), 'utf8')) as ClassifierArtifact;
const ruleSet = JSON.parse(readFileSync(join(dir, built.manifest.files.rules.key), 'utf8')) as RuleSet;
const ruleLabel = new Map(ruleSet.rules.map((r) => [r.id, r.label]));
/** The label a fired id stands for: a rule's label, or the edge's ('edge:<label>'). */
const labelOf = { get: (id: string) => ruleLabel.get(id) ?? (id.startsWith('edge:') ? id.slice(5) : undefined) };
const mode = built.enforcement.length ? 'shadow' : 'enforce';
const events: RouterEvent[] = [];
// The embedder behind the router is the training cache, warmed in batches: decisions are what is measured here.
const cache = await releaseEmbedderFor(artifact.embedding);
await cache.embedMany(future.map((d) => d.text));
const router = await loadRelease(dir, { mode, encoder: await cache.encoder(), monitoringRate: 1, observer: (e) => { events.push(e); } });
// Monitoring every rules-settled message gives every head's score, so each head's promise can be checked from telemetry.
const observed = await (async () => {
  const results: RouteResult[] = [];
  const t0 = performance.now();
  for (const d of future) results.push(await router.route({ text: d.text, requestId: d.id }));
  const ms = performance.now() - t0;
  await router.flush();
  return { results, ms };
})();
const decisionOf = new Map<string, Extract<RouterEvent, { type: 'decision' }>>();
const scoresOf = new Map<string, Readonly<Record<string, number>>>();
for (const e of events) {
  if (e.type === 'decision') { decisionOf.set(e.requestId, e); if (e.calibratedProbabilities) scoresOf.set(e.requestId, e.calibratedProbabilities); }
  if (e.type === 'diagnostic') scoresOf.set(e.requestId, e.calibratedProbabilities);
}
const missing = future.filter((d) => !decisionOf.has(d.id) || !scoresOf.has(d.id)).length;
if (missing) throw new Error(`telemetry is missing for ${missing} messages`);

// Each head on its own (its promise is per head): its rule or the edge fires it, a dismissal (rules or
// edge) clears it, else its threshold.
const fired: Record<string, boolean[]> = Object.fromEntries(heads.map((h) => [h, future.map((d) => {
  const ev = decisionOf.get(d.id)!;
  if ((ev.firedRule !== null && labelOf.get(ev.firedRule) === h) || (ev.edge?.fired.includes(h) && !ev.dismissed.includes(h))) return true;
  return !ev.dismissed.includes(h) && scoresOf.get(d.id)![h]! >= artifact.heads[h]!.threshold;
})]));
// The router's decisions against the always-embed reference evaluator on the same findings and scores.
const evaluator = createDecisionEvaluator({ heads: Object.fromEntries(heads.map((h) => [h, { threshold: artifact.heads[h]!.threshold, reviewFloor: artifact.heads[h]!.review_floor }])) }, routerPolicy(cfg));
const candidateOf = (r: RouteResult) => (r.outcome === 'shadow' ? r.candidate : r);
let mismatches = 0, routed = 0, routedRight = 0, shouldNot = 0, routedWrongly = 0, settledCount = 0, unavailable = 0;
const perRoute = Object.fromEntries(heads.map((h) => [h, { routed: 0, right: 0, support: 0, hit: 0 }]));
observed.results.forEach((r, i) => {
  const d = future[i], ev = decisionOf.get(d.id)!;
  if (r.outcome === 'unavailable') { unavailable++; return; }
  const c = candidateOf(r) as { outcome: string; routeId?: string };
  const ref = evaluator.decide({ fired: ev.firedRule ? { id: ev.firedRule, label: labelOf.get(ev.firedRule)!, source: ev.firedRule.startsWith('edge:') ? 'edge' : 'rule' } : null, dismissed: ev.dismissed }, scoresOf.get(d.id)!);
  if (ref.outcome !== c.outcome || (ref.outcome === 'route' && ref.routeId !== c.routeId)) mismatches++;
  if (ev.rulesSettled) settledCount++;
  const truth = heads.filter((h) => d.truth[h] === 1);
  if (!truth.length) shouldNot++;
  for (const h of truth) perRoute[h].support++;
  if (c.outcome === 'route') {
    routed++;
    perRoute[c.routeId!].routed++;
    if (truth.includes(c.routeId!)) { routedRight++; perRoute[c.routeId!].right++; perRoute[c.routeId!].hit++; } else if (!truth.length) routedWrongly++;
  }
});

const pct = (a: number, b: number) => (b ? `${(100 * a / b).toFixed(1)}%` : 'n/a');
const lines = [`# Release acceptance: ${cfg.name} (seed ${seed})`, '',
  `Future file: ${future.length} messages, fully labelled, each routed by @liquidau/router ${built.manifest.packages['@liquidau/router']} in ${mode} mode (release ${built.manifest.releaseId}, manifest ${built.manifestSha256.slice(0, 12)}). Gates passed: ${summary.gates_passed}; labels spent: ${summary.labels_spent}.${built.enforcement.length ? ` Shadow only: ${built.enforcement.join('; ')}.` : ''}`, '',
  '## Each head\'s promise', '',
  '| head | type | promise | met on the future? | recall | false alarms | precision | fires on |', '|---|---|---|---|---|---|---|---|'];
for (const h of cfg.heads) {
  let tp = 0, fp = 0, pos = 0;
  future.forEach((d, i) => { const y = d.truth[h.name]; pos += y; if (fired[h.name][i]) { if (y) tp++; else fp++; } });
  const recall = tp / pos, fa = fp / (future.length - pos), precision = tp + fp ? tp / (tp + fp) : NaN;
  const g = artifact.heads[h.name]!.guarantee;
  const promised = g && g.kind !== 'none';
  const met = !promised ? '-' : h.kind === 'recall' ? (recall >= h.target ? 'yes' : 'NO') : (precision >= h.target ? 'yes' : 'NO');
  const fireRate = fired[h.name].filter(Boolean).length / future.length;
  lines.push(`| ${h.name} | ${summary.heads[h.name].type} | ${promised ? `${h.kind} ≥ ${h.target} (${g!.kind})` : 'none (gates refused)'} | ${met} | ${recall.toFixed(3)} | ${(100 * fa).toFixed(2)}% | ${precision.toFixed(3)} | ${(100 * fireRate).toFixed(1)}%${h.maxFireRate !== undefined ? ` (cap ${100 * h.maxFireRate}%)` : ''} |`);
}
lines.push('', '## Final routing (after priority, suppression and dismissal)', '',
  `Coverage ${pct(routed, future.length)}; routed accuracy ${pct(routedRight, routed)}; false routes ${pct(routedWrongly, shouldNot)} of messages with no label; unavailable ${unavailable}.`, '',
  '| route | routed | precision | recall |', '|---|---|---|---|',
  ...heads.map((h) => `| ${h} | ${perRoute[h].routed} | ${pct(perRoute[h].right, perRoute[h].routed)} | ${pct(perRoute[h].hit, perRoute[h].support)} |`), '',
  `Edge tier (rules${summary.edge ? ' and the edge model' : ''}): settles ${pct(settledCount, future.length)} of the future file (frame estimate at release: ${(100 * summary.rules_tier_share_of_frame).toFixed(1)}%).`,
  `Router vs the always-embed reference evaluator: ${mismatches} mismatches in ${future.length - unavailable} decisions. Routing time ${(1000 * observed.ms / future.length).toFixed(0)} µs per message with cached embeddings.`);
writeFileSync(join(out, 'acceptance.md'), lines.join('\n'));
console.log(lines.join('\n'));
if (mismatches || unavailable) process.exitCode = 1;
