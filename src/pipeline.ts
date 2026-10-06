/**
 * Stages 1-3 of the example as one reusable step: data, rules, sampling and labelling for one
 * scenario seed. Seed 7 is the example's own run; other seeds vary the scenario, the pilot, the
 * main design, the queue and the simulated reviewer together, to repeat the run.
 */
import {
  applyReviews, designOf, designSample, labelQueue, requiredSampleSize, retrieveFromSeeds,
  type ExampleRecord, type FrameItem, type QueueItem, type Split,
} from '@liquidau/embedding-classifier';
import {
  buildRuleSet, certifyFalseAlarms, certifyPrefixes, evaluateFired, firingReport, hardNegativeItems, hitMatrix, mineRules, ruleSetMatcher, validateRules,
  type RecordLike,
} from '@liquidau/rule-miner';
import { decisionFunction, fitLogistic, seededRandom, sigmoid } from '@liquidau/solvers';

import { buildScenario, HEADS, type Head, type Message } from './data.ts';

export const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
export const lengthBucket = (text: string) => { const n = text.split(/\s+/).length; return n <= 6 ? 'short' : n <= 14 ? 'medium' : 'long'; };
export const topicOf = (intent: string) =>
  /card|pin|contactless|visa|apple_pay/.test(intent) ? 'card' : /transfer|beneficiary/.test(intent) ? 'transfer'
    : /top_up|topping/.test(intent) ? 'top_up' : /cash|atm/.test(intent) ? 'cash' : /exchange|currency|fiat/.test(intent) ? 'exchange'
      : /identity|verify|passcode|personal|account|age/.test(intent) ? 'account' : 'other';

export interface Context {
  messages: readonly Message[];
  byId: ReadonlyMap<string, Message>;
  /** Embedding of a message (the shipped model's). */
  emb: (m: { id: string }) => number[];
  log?: (line?: string) => void;
}

/**
 * How the main sample is drawn. Default: expected-positives allocation sized from the pilot, at least
 * 2,400 (the example's own run). `proportional` draws `total` with no pilot.
 */
/** A frame item as a custom stratifying score sees it. */
export interface Labelled { id: string; text: string; truth: Readonly<Record<string, 0 | 1>> }

export interface SampleOptions {
  method?: 'expected-positives' | 'proportional'; total?: number;
  /**
   * Optional: the score the frame is stratified by, in place of the round-0 head's. Fitted
   * from the first labelled round only (e.g. an offline teacher recalibrated on it), before sampling.
   */
  stratify?: { name: string; fit: (hist: readonly Labelled[]) => (m: Labelled) => number };
  /** Optional, with a proportional sample: strata in the top score band at this multiple of their share (same total). Only with an independent score (e.g. an offline teacher's). */
  topFactor?: number;
}

/**
 * A proportional sample with the top score band's strata weighted by `factor`,
 * as a manual allocation on the same strata: n_h = clamp(c · N_h · f_h, min(30, N_h), N_h), c the
 * largest that keeps the total.
 */
export function oversampledTop(options: Omit<Parameters<typeof designSample>[0], 'allocation' | 'seed'>, total: number, factor: number, seed: number) {
  const strata = designSample({ ...options, allocation: { total, method: 'proportional' }, seed }).design.strata;
  const f = strata.map((st) => (st.band[1] === Infinity ? factor : 1));
  const size = (c: number) => strata.map((st, i) => Math.min(st.N, Math.max(Math.min(30, st.N), Math.floor(c * st.N * f[i]))));
  let lo = 0, hi = 1;
  for (let k = 0; k < 50; k++) { const mid = (lo + hi) / 2; if (size(mid).reduce((a, b) => a + b, 0) <= total) lo = mid; else hi = mid; }
  const n = size(lo);
  return designSample({ ...options, allocation: { total, method: 'manual', manual: Object.fromEntries(strata.map((st, i) => [st.name, n[i]])) }, seed });
}

export async function buildRound(seed: number, ctx: Context, sampleOptions: SampleOptions = {}) {
  const { messages, byId, emb } = ctx;
  const log = ctx.log ?? (() => {});
  const section = (title: string) => log(`\n=== ${title} ===`);
  const offset = seed - 7;
  const scenario = buildScenario(messages, { seed, prevalence: { urgent: 0.04, refund: 0.02 }, historicalNegatives: 1500, sizes: { veto: 1000, certify: 1000, budget: 800, live: 1200 } });
  for (const [part, ms] of Object.entries(scenario) as Array<[string, Message[]]>) {
    log(`${part.padEnd(10)} ${String(ms.length).padStart(5)} messages, urgent ${pct(ms.filter((m) => m.truth.urgent).length / ms.length)}, refund ${pct(ms.filter((m) => m.truth.refund).length / ms.length)}`);
  }

  const traffic = (m: Message, backgroundUse: 'veto' | 'certify' | 'budget'): ExampleRecord & RecordLike =>
    ({ id: m.id, text: m.text, group: m.group, role: 'background', backgroundUse, source: { kind: 'traffic' }, labels: {} });
  const veto = scenario.veto.map((m) => traffic(m, 'veto'));
  const certify = scenario.certify.map((m) => traffic(m, 'certify'));
  const budget = scenario.budget.map((m) => traffic(m, 'budget'));

  // ---------------------------------------------------------------------------------------------
  section('1 rules (rule-miner)');
  // Historical labels: mine on 70%, validate on the rest.
  const hist = scenario.historical;
  const mineSet = hist.filter((_, i) => i % 10 < 7), validateSet = hist.filter((_, i) => i % 10 >= 7);
  const mined = mineRules(
    { texts: mineSet.map((m) => m.text), y: mineSet.map((m) => m.truth.urgent), groups: mineSet.map((m) => m.group), backgroundRecords: veto },
    // The veto background is real traffic, so it holds urgent messages too (~4%): a rule that fires on
    // them is doing its job. Allow a share well below prevalence.
    { minGroups: 3, maxBackgroundRate: 0.01, maxRules: 25 },
  );
  const { kept } = validateRules(mined, validateSet.map((m) => m.text), validateSet.map((m) => m.truth.urgent), { maxFalsePositives: 1 });
  // Firing-rate certification on unlabelled traffic counts correct catches too: at ~4% prevalence a 2%
  // budget is spent on them. Shown for contrast; the rules are certified on labelled
  // negatives in stage 3 instead.
  const firingCert = certifyPrefixes(hitMatrix(kept, certify.map((r) => r.text)), { maxRate: 0.02, records: certify });
  log(`mined ${mined.length} rules, ${kept.length} survive validation; firing-rate certification (<= 2% of traffic) would keep only ${firingCert.certified}`);
  // All validated rules are candidates until stage 3; they also define the sampling strata.
  const candidates = buildRuleSet('2026-10-03-candidates', [{ label: 'urgent', rules: kept }], { createdAt: '2026-10-03T00:00:00Z' });
  const candidateMatcher = ruleSetMatcher(candidates);
  log(`hard-negative briefs available for generation: ${hardNegativeItems(candidates.rules).length}`);

  // ---------------------------------------------------------------------------------------------
  section('2 sample (embedding-classifier + solvers)');
  // A round-0 model trained on historical labels scores the traffic - from text only, as designSample requires.
  const round0 = Object.fromEntries(HEADS.map((h) => [h, fitLogistic(hist.map(emb), hist.map((m) => m.truth[h]), { C: 1, classWeight: 'balanced' })])) as Record<Head, ReturnType<typeof fitLogistic>>;
  const round0Score = (m: Message, h: Head = 'urgent') => sigmoid(decisionFunction(round0[h], emb(m)));
  const stratScore = sampleOptions.stratify?.fit(hist) ?? ((m: Message) => round0Score(m));
  const frame: FrameItem[] = scenario.frame.map((m) => ({ id: m.id, text: m.text, group: m.group, signals: { ruleFires: candidateMatcher.match(m.text) !== null, score: stratScore(m), slice: lengthBucket(m.text) } }));
  // Training data comes mostly from the historical set, so the sample favours calibration and test.
  const designOptions = { frame, scoreBands: [0.9, 0.7], scoringModel: sampleOptions.stratify?.name ?? 'round0-logistic-minilm', mergeSmallStrata: true, roleSplit: { train: 0.2, calibration: 0.4, test: 0.4 } } as const;

  // Pilot round (proportional, labelled): per-stratum positive rates that size the main sample.
  const pilot = designSample({ ...designOptions, allocation: { total: 600, method: 'proportional' }, seed: 101 + offset });
  const priors = Object.fromEntries(HEADS.map((h) => [h, {} as Record<string, { positives: number; labelled: number }>]));
  for (const r of pilot.records) {
    const s = (r.source as { stratum: string }).stratum;
    for (const h of HEADS) {
      priors[h][s] ??= { positives: 0, labelled: 0 };
      priors[h][s].labelled++;
      priors[h][s].positives += byId.get(r.id)!.truth[h];
    }
  }
  for (const h of HEADS) log(`pilot ${h}: ${Object.entries(priors[h]).map(([s, c]) => `${s} ${c.positives}/${c.labelled}`).join(', ')}`);
  // Size the main sample so every stratum holding a material share of either head's positives expects 6
  // in calibration (in simulation ~6 was enough for design-based recall bounds to hold).
  const sizing = { priors, minExpectedPositives: 6 };
  const required = requiredSampleSize({ ...designOptions, allocation: sizing });
  const proportional = sampleOptions.method === 'proportional';
  const total = Math.min(frame.length, proportional ? sampleOptions.total ?? 2400 : Math.max(required, sampleOptions.total ?? 2400));
  const main = proportional && sampleOptions.topFactor && sampleOptions.topFactor !== 1
    ? oversampledTop(designOptions, total, sampleOptions.topFactor, 202 + offset)
    : designSample({ ...designOptions, allocation: proportional ? { total, method: 'proportional' } : { total, method: 'expected-positives', ...sizing }, seed: 202 + offset });
  log(`main design ${main.designId}: ${main.records.length} of ${frame.length} frame messages (expected-positives sizing asked for ${required})`);
  for (const s of main.design.strata) log(`  ${s.name.padEnd(28)} N ${String(s.N).padStart(5)}  n ${String(s.n).padStart(4)}  calibration ${String(s.roles.calibration).padStart(3)}  E[positives] ${HEADS.map((h) => `${h} ${((main.design.allocation.priorRatesByHead?.[h]?.[s.name] ?? 0) * s.roles.calibration).toFixed(1)}`).join(', ')}`);
  for (const m of main.design.merged) log(`  merged: ${m.name} <- ${m.from.join(', ')}`);
  for (const w of main.design.warnings) log(`  warning: ${w}`);

  // ---------------------------------------------------------------------------------------------
  section('3 label (blinded queue, simulated reviewer)');
  const sampledGroups = new Set(main.records.map((r) => r.group));
  const reserved = new Set([...sampledGroups, ...[...veto, ...certify, ...budget].map((r) => r.group), ...scenario.live.map((m) => m.group), ...hist.map((m) => m.group)]);
  const pool = scenario.frame.filter((m) => !reserved.has(m.group)).map((m) => ({ record: { id: m.id, text: m.text, group: m.group, role: 'train' as const, source: { kind: 'traffic' as const }, labels: {} }, embedding: emb(m) }));
  const seeds = hist.filter((m) => m.truth.urgent).slice(0, 40).map((m) => ({ record: { id: m.id, text: m.text, group: m.group, role: 'train' as const, source: { kind: 'traffic' as const }, labels: { urgent: 1 as const }, labelledBy: 'human' as const }, embedding: emb(m) }));
  const evaluationEmbeddings = main.records.filter((r) => r.role !== 'train').map(emb);
  const retrieved = await retrieveFromSeeds({ head: 'urgent', seeds, pool, evaluation: evaluationEmbeddings, candidates: 50, k: 5, floor: 0.6, round: 1, reservedGroups: reserved });
  log(`retrieval: ${retrieved.length} candidates near ${seeds.length} historical urgent messages`);
  const items: QueueItem[] = [
    ...main.records.map((record) => ({ record, mechanism: 'sampled' as const, heads: [...HEADS] })),
    ...retrieved.map((record) => ({ record, mechanism: 'retrieved' as const, heads: ['urgent'] })),
  ];
  const queue = labelQueue({ items, budget: { sampled: main.records.length, retrieved: 150, verification: 0 }, seed: 303 + offset });
  log(`queue ${queue.queueId}: ${queue.review.length} items for reviewers (fields shown: ${Object.keys(queue.review[0]).join(', ')})`);
  // The simulated reviewer answers from gold labels and skips 2% of items at random.
  const reviewer = seededRandom(404 + offset);
  const reviews = queue.review.filter(() => reviewer() >= 0.02).map((item) => {
    const m = byId.get(queue.key.items[item.itemId].record.id)!;
    return { itemId: item.itemId, labels: Object.fromEntries(item.heads.map((h) => [h, m.truth[h as Head]])) as Record<string, 0 | 1> };
  });
  const labelled = applyReviews(queue.key, reviews);
  log(`labelled ${labelled.records.length}; ${labelled.warnings.length} warning(s)`);
  for (const w of labelled.warnings.slice(0, 5)) log(`  ${w}`);
  const confirmed = labelled.records.filter((r) => r.source.kind === 'retrieved');
  log(`retrieval hit rate: ${confirmed.filter((r) => r.labels.urgent === 1).length}/${confirmed.length} confirmed urgent`);

  // Certify the rules on the labelled calibration sample's NEGATIVES, weighted by its design: a
  // false-alarm statement, so correct catches cost nothing.
  const sampledFor = (role: 'calibration' | 'test') => labelled.records.filter((r) => r.role === role && r.source.kind === 'sampled' && (r.labels.urgent === 0 || r.labels.urgent === 1));
  const calRecords = sampledFor('calibration');
  // Rules are deterministic, so their firing on the whole frame is known without labels: it caps each
  // stratum's possible false alarms exactly (zero where they never fire).
  const framed = scenario.frame.filter((m) => main.stratumOf[m.id] !== undefined);
  const faCert = certifyFalseAlarms(hitMatrix(kept, calRecords.map((r) => r.text)), {
    y: calRecords.map((r) => r.labels.urgent as 0 | 1), maxRate: 0.01, design: designOf(calRecords),
    frame: { strata: framed.map((m) => main.stratumOf[m.id]), fired: hitMatrix(kept, framed.map((m) => m.text)) },
  });
  const ruleSet = buildRuleSet('2026-10-03', [{ label: 'urgent', rules: kept.slice(0, faCert.certified) }], { createdAt: '2026-10-03T00:00:00Z' });
  const matcher = ruleSetMatcher(ruleSet);
  log(`rules: ${faCert.certified} of ${kept.length} certified to fire on <= 1% of negatives (${faCert.method}, 95%) - firing-rate certification kept ${firingCert.certified}`);
  for (const r of ruleSet.rules.slice(0, 6)) log(`  ${r.id.padEnd(12)} ${r.regex.join(' & ')}`);
  // Their recall and false alarms on the test sample, weighted by its design.
  const testRecords = sampledFor('test');
  const ruleEval = evaluateFired(testRecords.map((r) => matcher.match(r.text) !== null), testRecords.map((r) => r.labels.urgent as 0 | 1), { design: designOf(testRecords) });
  log(`rules on test (design-weighted): recall ${ruleEval.recall.toFixed(3)} [${ruleEval.recall_ci95.map((x) => x.toFixed(3)).join(', ')}], false alarms ${pct(ruleEval.false_alarm_rate)}; on live traffic they fire on ${pct(firingReport(ruleSet, scenario.live.map((m) => m.text)).rate)}`);
  const historicalRecords: ExampleRecord[] = hist.map((m) => ({ id: m.id, text: m.text, group: m.group, role: 'train', source: { kind: 'traffic' }, labels: { urgent: m.truth.urgent, refund: m.truth.refund }, labelledBy: 'human' }));
  const records: ExampleRecord[] = [...historicalRecords, ...labelled.records];
  const split = records.map((r) => r.role as Split);
  const label = (h: Head) => records.map((r) => (r.labels[h] === 0 || r.labels[h] === 1 ? r.labels[h] : null));
  // Labels spent on this round's sampling: the main sample, plus the pilot when it sized the sample.
  const labelsSpent = main.records.length + (proportional ? 0 : pilot.records.length);
  return {
    scenario, veto, certify, budget, hist, kept, firingCert, candidates, main, pool, labelled, faCert, ruleSet, matcher, ruleEval, labelsSpent,
    records, split, label,
  };
}

export type Round = Awaited<ReturnType<typeof buildRound>>;
