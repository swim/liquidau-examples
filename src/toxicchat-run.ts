/**
 * The same loop on ToxicChat: real prompts at their natural prevalence, two rare heads.
 *
 *   0  benchmark     plain logistic regression on the official split, against the dataset's published
 *                    baselines (average precision, threshold-free)
 *   1  rules         rule-miner mines `toxic` rules from a first labelled round, vetoed by traffic
 *   2  sample        a round-0 model scores the rest of the traffic; a stratified probability sample
 *   3  label         blinded queue, simulated reviewer; rules certified on labelled negatives
 *   4  train         trainHeads with design-based recall guarantees for both heads
 *   5  production    the official test file is a fully labelled later period: did each guarantee hold?
 *                    Compared with a heuristic threshold and with the rules
 *   6  monitor       drift checks on that period against the budget traffic
 *
 * Unlike Banking77, step 5 checks guarantees against every label in a later period, not a sample.
 * Seeded: rerunning reproduces the same results. Writes out/toxicchat.md.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { monitorWindow, scoreEmbedding, type ClassifierArtifact, type TrainResult } from '@liquidau/embedding-classifier';
import { evaluateFired } from '@liquidau/rule-miner';
import { decisionFunction, fitLogistic, seededRandom } from '@liquidau/solvers';

import { EMBEDDING, makeEmbedder } from './embed.ts';
import { pct } from './pipeline.ts';
import { buildToxicRound, POLICIES, rate, TARGET } from './toxicchat-round.ts';
import { loadToxicChat, MAX_CHARS, TC_HEADS, type Prompt, type TcHead } from './toxicchat.ts';

const OUT = join(import.meta.dirname, '..', 'out');
const lines: string[] = [];
const log = (line = '') => { console.log(line); lines.push(line); };
const section = (title: string) => log(`\n=== ${title} ===`);

/** Average precision as scikit-learn computes it (step-wise, no interpolation). */
function averagePrecision(scores: readonly number[], y: readonly (0 | 1)[]): number {
  const order = scores.map((s, i) => [s, y[i]] as const).sort((a, b) => b[0] - a[0]);
  const positives = y.reduce<number>((a, b) => a + b, 0);
  let tp = 0, fp = 0, ap = 0, prevRecall = 0;
  for (let i = 0; i < order.length; i++) {
    if (order[i][1]) tp++; else fp++;
    if (i + 1 < order.length && order[i + 1][0] === order[i][0]) continue; // ties form one step
    const recall = tp / positives;
    ap += (recall - prevRecall) * (tp / (tp + fp));
    prevRecall = recall;
  }
  return ap;
}

// -----------------------------------------------------------------------------------------------
section('data');
const prompts = await loadToxicChat();
const byId = new Map(prompts.map((p) => [p.id, p]));
const embedder = await makeEmbedder(() => {}, undefined, { cache: 'toxicchat', maxChars: MAX_CHARS });
const vectors = await embedder.embedMany(prompts.map((p) => p.text));
const embeddingOf = new Map(prompts.map((p, i) => [p.id, vectors[i]]));
const emb = (m: { id: string }) => embeddingOf.get(m.id)!;
const trainFile = prompts.filter((p) => p.split === 'train'), testFile = prompts.filter((p) => p.split === 'test');
log(`${prompts.length} prompts embedded with ${EMBEDDING.model_id} (truncated to ${MAX_CHARS} characters)`);

// -----------------------------------------------------------------------------------------------
section('0 benchmark (official split, all labels)');
const published: Record<TcHead, string> = { toxic: 'ToxicChat-T5-large 0.885, OpenAI Moderation 0.631 (dataset card, version 0124)', jailbreak: 'none published' };
for (const h of TC_HEADS) {
  const model = fitLogistic(trainFile.map(emb), trainFile.map((p) => p.truth[h]), { C: 1, classWeight: 'balanced' });
  const s = testFile.map((p) => decisionFunction(model, emb(p)));
  log(`${h}: average precision ${averagePrecision(s, testFile.map((p) => p.truth[h])).toFixed(3)} on ${testFile.length} test prompts; published: ${published[h]}`);
}

const { budgetSet, matcher, train } = buildToxicRound(7, { trainFile, byId, emb, log });

// -----------------------------------------------------------------------------------------------
section('4 train (embedding-classifier)');
const result = train(POLICIES.linearised, { log: (l) => log(`  ${l}`) });
for (const h of TC_HEADS) {
  const spec = result.heads[h], d = result.evaluation[h]?.design;
  if (!spec || !d) { log(`${h}: no head (${result.failures.filter((f) => f.startsWith(h)).join('; ')})`); continue; }
  log(`${h}: threshold ${spec.threshold.toFixed(4)}, guarantee ${spec.guarantee?.kind ?? 'none'}; sampled test: recall ${d.recall.estimate.toFixed(3)} [${d.recall.ci95.map((x) => x.toFixed(3)).join(', ')}], false alarms ${pct(d.false_alarm_rate.estimate)}, effective positives ${d.effective_positives.toFixed(1)}`);
}
log(`gate failures: ${result.failures.length ? '' : 'none'}`);
for (const f of result.failures) log(`  FAIL ${f}`);
for (const w of result.warnings.slice(0, 6)) log(`  warn ${w}`);
const artifactOf = (r: TrainResult<TcHead>): ClassifierArtifact<TcHead> => ({ version: '2026-10-03', created_at: '2026-10-03T00:00:00Z', embedding: { ...EMBEDDING }, heads: r.heads });
const artifact = artifactOf(result);

// -----------------------------------------------------------------------------------------------
section('5 production (official test file, every label)');
// Test prompts whose exact text also occurs in the train file are left out (230 groups).
const seen = new Set(trainFile.map((p) => p.group));
const production = testFile.filter((p) => !seen.has(p.group));
log(`${production.length} prompts (toxic ${rate(production, 'toxic')}, jailbreak ${rate(production, 'jailbreak')})`);
const rows = ['| head | approach | stated guarantee | recall | precision | false alarms | fires on |', '|---|---|---|---|---|---|---|'];
const row = (h: TcHead, name: string, guarantee: string, fired: boolean[]) => {
  const y = production.map((p) => p.truth[h]);
  const ev = evaluateFired(fired, y);
  const tp = fired.filter((f, i) => f && y[i]).length, n = fired.filter(Boolean).length;
  rows.push(`| ${h} | ${name} | ${guarantee} | ${ev.recall.toFixed(3)} (${ev.recall_ci95.map((x) => x.toFixed(3)).join('–')}) | ${n ? (tp / n).toFixed(3) : 'n/a'} | ${pct(ev.false_alarm_rate)} | ${pct(n / fired.length)} |`);
};
const heuristic = train(POLICIES.heuristic);
const exact = train(POLICIES.exact);
for (const h of TC_HEADS) if (exact.failures.some((f) => f.startsWith(h))) log(`exact design bound, ${h}: no head (${exact.failures.filter((f) => f.startsWith(h)).join('; ')})`);
const classifierFired = (r: TrainResult<TcHead>, h: TcHead) => production.map((p) => scoreEmbedding(artifactOf(r), emb(p))[h]! >= r.heads[h]!.threshold);
const stated = (r: TrainResult<TcHead>, h: TcHead) => {
  const g = r.heads[h]?.guarantee;
  const failed = r.failures.some((f) => f.startsWith(`${h}:`)) ? '; failed its gates' : '';
  return (g && g.kind !== 'none' ? `recall ≥ ${TARGET[h]} (${g.kind}, 95%)` : 'none') + failed;
};
const rulesFired = production.map((p) => matcher.match(p.text) !== null);
row('toxic', 'certified rules only', 'false alarms ≤ 1% (95%)', rulesFired);
for (const h of TC_HEADS) {
  for (const [name, r] of [['design threshold, linearised (shipped)', result], ['design threshold, exact', exact], ['heuristic threshold', heuristic]] as const) {
    if (r.heads[h]) row(h, name, stated(r, h), classifierFired(r, h));
  }
}
if (result.heads.toxic) row('toxic', 'rules or classifier', '-', classifierFired(result, 'toxic').map((f, i) => f || rulesFired[i]));
for (const r of rows) log(r);
if (result.heads.toxic) {
  // The 230 test prompts that also occur in the train file, scored the same way (left out above).
  const dup = testFile.filter((p) => seen.has(p.group) && p.truth.toxic);
  log(`toxic prompts also in the train file: ${dup.length}, of which the shipped threshold catches ${dup.filter((p) => scoreEmbedding(artifact, emb(p)).toxic! >= result.heads.toxic!.threshold).length}`);
}

// -----------------------------------------------------------------------------------------------
section('6 monitor');
if (result.heads.toxic) {
  const score = (p: Prompt) => scoreEmbedding(artifact, emb(p)).toxic!;
  const check = (name: string, window: Prompt[]) => {
    const res = monitorWindow({ spec: artifact.heads.toxic!, firingBound: result.evaluation.toxic!.background_rate_upper, reference: budgetSet.map(score), live: window.map(score), alpha: 0.01 });
    log(`${name} (toxic ${rate(window, 'toxic')}): ${res.alert ? 'ALERT' : 'ok'}`);
    for (const c of res.checks) log(`    ${c.test.padEnd(12)} p ${c.pValue.toExponential(1)}  ${c.detail}`);
  };
  // A random window of the production period, then its first 1,200 rows: the file is ordered, and
  // those rows (all human-annotated) are about 3x as toxic as the rest - a real shift to detect.
  const pick = seededRandom(505);
  check('random window, 1,200 prompts', production.map((p) => [pick(), p] as const).sort((a, b) => a[0] - b[0]).slice(0, 1200).map(([, p]) => p));
  check('first 1,200 rows of the test file', production.slice(0, 1200));
}

mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, 'toxicchat.md'), ['# ToxicChat run', '', 'WARNING: the data behind these numbers contains offensive text.', '', '```', ...lines, '```', ''].join('\n'));
console.log('\nwrote out/toxicchat.md');
