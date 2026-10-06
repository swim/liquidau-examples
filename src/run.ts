/**
 * The whole loop on Banking77, end to end:
 *
 *   1  rules         rule-miner mines rules for `urgent` from historical labels, vetoed by ordinary
 *                    traffic; once the sample is labelled (3) they are certified on its negatives
 *   2  sample        a round-0 model scores the traffic; embedding-classifier stratifies it by rule
 *                    firing × score band and draws a probability sample, sized from a pilot round
 *   3  label         one blinded queue for sampled messages and retrieval candidates near known
 *                    positives; a simulated reviewer answers from Banking77's gold labels
 *   4  train         trainHeads with provenance rules, design-weighted calibration, a design-based
 *                    recall guarantee for `urgent` and precision guarantee for `refund`, the rules as
 *                    baseline, per-slice and calibration gates
 *   5  ship          artifact, round-trip check, report with coverage, a decision demo
 *   6  next round    disagreement queues between rules and classifier
 *   7  monitor       drift checks on a live window, then on one with a fraud wave injected
 *   8  compare       the same split against baselines (rules only, a heuristic threshold, a stronger
 *                    embedding model), and where `urgent`'s false alarms and misses come from
 *
 * Everything is seeded: rerunning reproduces the same results.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  assertRoundTrip, coverageReport, decide, defineAxes, designOf, monitorWindow,
  publishPlan, reportMarkdown, scoreEmbedding, trainHeads,
  type ClassifierArtifact, type HeadPolicy, type TrainResult,
} from '@liquidau/embedding-classifier';
import { disagreementQueues, evaluateFired, ruleSetHash } from '@liquidau/rule-miner';
import { stratifiedRatio } from '@liquidau/solvers';

import { HEADS, loadBanking77, type Head, type Message } from './data.ts';
import { EMBEDDING, makeEmbedder, SMALL_MODEL } from './embed.ts';
import { buildRound, lengthBucket, pct, topicOf } from './pipeline.ts';

const OUT = join(import.meta.dirname, '..', 'out');
const log = (line = '') => console.log(line);
const section = (title: string) => log(`\n=== ${title} ===`);
// ---------------------------------------------------------------------------------------------
section('data');
const messages = await loadBanking77();
const byId = new Map(messages.map((m) => [m.id, m]));
const embedder = await makeEmbedder();
const allTexts = messages.map((m) => m.text);
const vectors = await embedder.embedMany(allTexts);
const embeddingOf = new Map(messages.map((m, i) => [m.id, vectors[i]]));
const emb = (m: { id: string }) => embeddingOf.get(m.id)!;
log(`embedded ${vectors.length} messages with ${EMBEDDING.model_id}`);
const { scenario, budget, hist, main, pool, labelled, ruleSet, matcher, records, split, label } = await buildRound(7, { messages, byId, emb, log });
const X = records.map(emb);

// ---------------------------------------------------------------------------------------------
section('4 train (embedding-classifier)');
const POLICIES: Record<Head, HeadPolicy> = {
  urgent: { kind: 'recall', targetRecall: 0.85, mode: 'design', designMethod: 'linearised', delta: 0.05, sliceGate: { minPositives: 10 } },
  // About 18 calibration positives can't certify more than 0.7: a data limit, not a defect.
  refund: { kind: 'precision', targetPrecision: 0.7, mode: 'design', delta: 0.05 },
};
/** One training run on the shared records and splits; embeddings and policies can vary (stage 8). */
const train = (embedOf: (m: { id: string }) => number[], policies: Record<Head, HeadPolicy>, quiet = false) => trainHeads({
  X: records.map(embedOf), split, records, designs: [main.design], seed: 1, calibrationAlpha: 0.01,
  slices: { length: records.map((r) => lengthBucket(r.text)) },
  // Budget traffic holds urgent messages at ~4%, so its firing budget must sit above prevalence × recall.
  background: { X: budget.map(embedOf), records: budget, maxRate: { urgent: 0.08, refund: 0.04 } },
  heads: [
    // Linear heads: stage 8 compares embedding models and thresholds on the same head type.
    { name: 'urgent', y: label('urgent'), baseline: records.map((r) => matcher.match(r.text) !== null), policy: policies.urgent, type: 'linear' },
    { name: 'refund', y: label('refund'), policy: policies.refund, type: 'linear' },
  ],
  log: quiet ? () => {} : (l) => log(`  ${l}`),
});
const result = train(emb, POLICIES);
for (const h of HEADS) {
  const ev = result.evaluation[h]!, spec = result.heads[h]!, d = ev.design!;
  log(`${h}: threshold ${spec.threshold.toFixed(4)}, guarantee ${spec.guarantee?.kind ?? 'none'}`);
  log(`  recall ${d.recall.estimate.toFixed(3)} [${d.recall.ci95.map((x) => x.toFixed(3)).join(', ')}], precision ${d.precision?.estimate.toFixed(3) ?? 'n/a'}, false alarms ${pct(d.false_alarm_rate.estimate)}, prevalence ${pct(d.prevalence.estimate)}, effective positives ${d.effective_positives.toFixed(1)}`);
  if (ev.vs_baseline) log(`  vs rules: ${JSON.stringify(ev.vs_baseline)}`);
  if (ev.calibration_test) log(`  calibration (Cox): intercept ${ev.calibration_test.intercept.toFixed(2)}, slope ${ev.calibration_test.slope.toFixed(2)}, p ${ev.calibration_test.p_value.toExponential(1)}`);
}
log(`gate failures: ${result.failures.length ? '' : 'none'}`);
for (const f of result.failures) log(`  FAIL ${f}`);
for (const w of result.warnings.slice(0, 8)) log(`  warn ${w}`);
if (result.warnings.length > 8) log(`  ... ${result.warnings.length - 8} more warnings in the report`);

// ---------------------------------------------------------------------------------------------
section('5 ship');
const artifact: ClassifierArtifact<Head> = {
  version: '2026-10-03', created_at: '2026-10-03T00:00:00Z', embedding: EMBEDDING, heads: result.heads, evaluation: result.evaluation,
  training: { design: result.design, provenance: result.provenance, rule_set: { version: ruleSet.version, hash: ruleSetHash(ruleSet) } },
  gates: { passed: result.failures.length === 0, failures: result.failures, warnings: result.warnings },
};
assertRoundTrip(artifact, X, result.testProbabilities);
const tags = Object.fromEntries(records.map((r) => [r.id, { length: lengthBucket(r.text), topic: topicOf(byId.get(r.id)!.intent) }]));
const coverage = coverageReport({ head: 'urgent', records, tags, axes: defineAxes({ length: { values: ['short', 'medium', 'long'], observable: true }, topic: { values: ['card', 'transfer', 'top_up', 'cash', 'exchange', 'account', 'other'], observable: false } }) });
mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, 'artifact.json'), JSON.stringify(artifact));
writeFileSync(join(OUT, 'ruleset.json'), JSON.stringify(ruleSet, null, 2));
writeFileSync(join(OUT, 'report.md'), reportMarkdown(artifact, { title: 'Banking77 support example', baselineName: 'rules', coverage }));
log(`wrote out/artifact.json, out/ruleset.json, out/report.md; publish plan: ${JSON.stringify(publishPlan({ gatesPassed: artifact.gates!.passed, shadowCandidate: !artifact.gates!.passed, promote: artifact.gates!.passed, generated: result.provenance?.generated }))}`);
for (const text of ['I think someone has stolen my card and used it', 'Can I get my money back for this order?', 'What is the exchange rate today?']) {
  const vec = (await embedder.embedMany([text]))[0];
  const d = decide(artifact, scoreEmbedding(artifact, vec), { priority: ['urgent', 'refund'] });
  log(`  "${text}" -> ${d.head ?? 'none'} (${d.reason}); rule: ${matcher.match(text)?.id ?? 'none'}`);
}

// ---------------------------------------------------------------------------------------------
section('6 next round (rule-miner)');
const unlabelled = pool.filter((p) => !labelled.records.some((r) => r.id === p.record.id)).map((p) => byId.get(p.record.id)!);
const scores = unlabelled.map((m) => scoreEmbedding(artifact, emb(m)).urgent!);
const queues = disagreementQueues(ruleSet, unlabelled.map((m) => m.text), scores, { reviewFloor: artifact.heads.urgent!.review_floor, threshold: artifact.heads.urgent!.threshold, label: 'urgent', limit: 50 });
for (const [q, list] of Object.entries(queues) as Array<[string, Array<{ index: number }>]>) log(`${q.padEnd(16)} ${list.length} (truly urgent: ${list.filter((it) => unlabelled[it.index].truth.urgent).length})`);

// ---------------------------------------------------------------------------------------------
section('7 monitor (embedding-classifier)');
const urgentScore = (m: Message) => scoreEmbedding(artifact, emb(m)).urgent!;
const reference = scenario.budget.map(urgentScore);
const firingBound = result.evaluation.urgent!.background_rate_upper;
const report = (name: string, live: number[]) => {
  const res = monitorWindow({ spec: artifact.heads.urgent!, firingBound, reference, live, alpha: 0.01 });
  log(`${name.padEnd(28)} ${res.alert ? 'ALERT' : 'ok   '} ${res.checks.filter((c) => c.alert).map((c) => c.test).join(', ')}`);
  for (const c of res.checks) log(`    ${c.test.padEnd(12)} p ${c.pValue.toExponential(1)}  ${c.detail}`);
};
report('live window', scenario.live.map(urgentScore));
// A fraud wave: 150 extra urgent messages (from the historical set) land in the same window.
const wave = [...scenario.live, ...hist.filter((m) => m.truth.urgent).slice(100, 250)];
report('live window + fraud wave', wave.map(urgentScore));

// ---------------------------------------------------------------------------------------------
section('8 compare: baselines and error analysis');
// Every row is measured on the same sampled test records, weighted by their design.
const testOf = (h: Head) => records.filter((r) => r.role === 'test' && (r.labels[h] === 0 || r.labels[h] === 1));
const measure = (h: Head, fired: boolean[]) => {
  const tr = testOf(h), y = tr.map((r) => r.labels[h] as 0 | 1), design = designOf(tr);
  const ev = evaluateFired(fired, y, { design });
  const anyFired = fired.some(Boolean);
  const precision = anyFired ? stratifiedRatio({ ...design, num: y.map((v, i) => v * Number(fired[i])), den: fired.map(Number) }).estimate : NaN;
  return { recall: ev.recall, recallCi: ev.recall_ci95, precision, falseAlarms: ev.false_alarm_rate };
};
const classifierFired = (r: TrainResult<Head>, h: Head) => {
  const p = new Map(r.testProbabilities[h]!.idx.map((i, j) => [records[i].id, r.testProbabilities[h]!.p[j]]));
  return testOf(h).map((rec) => p.get(rec.id)! >= r.heads[h]!.threshold);
};
const smallEmbedder = await makeEmbedder(() => {}, SMALL_MODEL);
const smallVectors = await smallEmbedder.embedMany(allTexts);
const smallOf = new Map(messages.map((m, i) => [m.id, smallVectors[i]]));
const heuristic: Record<Head, HeadPolicy> = {
  urgent: { kind: 'recall', targetRecall: 0.85, designRecall: 0.9, mode: 'heuristic' },
  refund: { kind: 'precision', targetPrecision: 0.7, mode: 'heuristic' },
};
const variants: Array<{ name: string; run: TrainResult<Head> }> = [
  { name: `mpnet, design threshold (shipped)`, run: result },
  { name: `mpnet, heuristic threshold`, run: train(emb, heuristic, true) },
  { name: `MiniLM, design threshold`, run: train((m) => smallOf.get(m.id)!, POLICIES, true) },
];
const rows: string[] = ['| head | approach | guarantee | recall (95% CI) | precision | false alarms |', '|---|---|---|---|---|---|'];
const fmtRow = (h: Head, name: string, guarantee: string, m: ReturnType<typeof measure>) =>
  `| ${h} | ${name} | ${guarantee} | ${m.recall.toFixed(3)} (${m.recallCi.map((x) => x.toFixed(3)).join('–')}) | ${Number.isNaN(m.precision) ? 'n/a' : m.precision.toFixed(3)} | ${pct(m.falseAlarms)} |`;
rows.push(fmtRow('urgent', 'certified rules only', `false alarms ≤ 1% of negatives`, measure('urgent', testOf('urgent').map((r) => matcher.match(r.text) !== null))));
for (const h of HEADS) for (const v of variants) {
  const g = v.run.heads[h]!.guarantee;
  const stated = g && g.kind !== 'none' ? `${g.metric ?? 'recall'} ≥ ${(1 - g.alpha).toFixed(2)} (${g.kind})` : v.run.failures.some((f) => f.startsWith(`${h}:`)) ? 'failed its gates' : 'none';
  rows.push(fmtRow(h, v.name, stated, measure(h, classifierFired(v.run, h))));
}
for (const row of rows) log(row);

// Where urgent's errors come from, by Banking77 intent, weighted by the test design (share of the weighted total).
const tr = testOf('urgent'), w = designOf(tr).inclusionProbs.map((p) => 1 / p), fired = classifierFired(result, 'urgent');
const breakdown = (pick: (i: number) => boolean) => {
  const by = new Map<string, number>();
  let total = 0;
  tr.forEach((r, i) => { if (pick(i)) { const intent = byId.get(r.id)!.intent; by.set(intent, (by.get(intent) ?? 0) + w[i]); total += w[i]; } });
  return [...by.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([intent, x]) => `${intent} ${pct(x / total)}`);
};
const falseAlarmsBy = breakdown((i) => fired[i] && tr[i].labels.urgent === 0);
const missesBy = breakdown((i) => !fired[i] && tr[i].labels.urgent === 1);
log(`urgent false alarms by intent: ${falseAlarmsBy.join(', ')}`);
log(`urgent misses by intent: ${missesBy.join(', ') || 'none'}`);
writeFileSync(join(OUT, 'comparison.md'), [
  '# Baselines and error analysis', '', 'Same split, same sampled test records; every rate is weighted by the test design.', '', ...rows, '',
  `**Urgent false alarms by intent** (share of weighted false alarms): ${falseAlarmsBy.join(', ')}`, '',
  `**Urgent misses by intent:** ${missesBy.join(', ') || 'none'}`, '',
].join('\n'));
log('wrote out/comparison.md');
