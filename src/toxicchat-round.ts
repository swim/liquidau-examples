/**
 * Stages 1-3 of the ToxicChat run as one seeded step (rules, sample, label), plus the shared
 * training call. Seed 7 is the run in toxicchat-run.ts; other seeds vary the split of the train
 * file into roles, the sample, the queue and the simulated reviewer together.
 */
import {
  applyReviews, designOf, designSample, labelQueue, requiredSampleSize, trainHeads,
  type ExampleRecord, type FrameItem, type HeadPolicy, type Split,
} from '@liquidau/embedding-classifier';
import { buildRuleSet, certifyFalseAlarms, hitMatrix, mineRules, ruleSetMatcher, validateRules, type RecordLike } from '@liquidau/rule-miner';
import { decisionFunction, fitLogistic, seededRandom, sigmoid } from '@liquidau/solvers';

import { oversampledTop, pct, type SampleOptions } from './pipeline.ts';
import { TC_HEADS, type Prompt, type TcHead } from './toxicchat.ts';

export const lengthBucket = (text: string) => { const n = text.split(/\s+/).length; return n <= 8 ? 'short' : n <= 40 ? 'medium' : 'long'; };
export const rate = (ps: readonly Prompt[], h: TcHead) => pct(ps.filter((p) => p.truth[h]).length / ps.length);

/** Recall each head must reach. */
export const TARGET: Record<TcHead, number> = { toxic: 0.8, jailbreak: 0.7 };
export const POLICIES = {
  linearised: (h: TcHead): HeadPolicy => ({ kind: 'recall', targetRecall: TARGET[h], mode: 'design', designMethod: 'linearised', delta: 0.05 }),
  exact: (h: TcHead): HeadPolicy => ({ kind: 'recall', targetRecall: TARGET[h], mode: 'design', designMethod: 'exact', delta: 0.05 }),
  heuristic: (h: TcHead): HeadPolicy => ({ kind: 'recall', targetRecall: TARGET[h], designRecall: TARGET[h] + 0.05, mode: 'heuristic' }),
};

export interface TcContext {
  /** The official train file: the traffic. */
  trainFile: readonly Prompt[];
  byId: ReadonlyMap<string, Prompt>;
  emb: (m: { id: string }) => number[];
  log?: (line?: string) => void;
}

/**
 * How the main sample is drawn. Default: proportional, 1,800 (the example's own run).
 * `expected-positives` sizes the sample from a 300-prompt proportional pilot so every stratum holding
 * a material share of either head's positives expects 6 calibration positives, at least `total`.
 */
export interface TcSampleOptions extends Pick<SampleOptions, 'stratify' | 'topFactor'> { method?: 'proportional' | 'expected-positives'; total?: number }

export function buildToxicRound(seed: number, ctx: TcContext, sampleOptions: TcSampleOptions = {}) {
  const { trainFile, byId, emb } = ctx;
  const log = ctx.log ?? (() => {});
  const section = (title: string) => log(`\n=== ${title} ===`);
  const offset = seed - 7;

  // ---------------------------------------------------------------------------------------------
  section('1 rules (rule-miner)');
  // The train file is the traffic. Whole groups go to one role: a first labelled round (random, so at
  // natural prevalence), unlabelled background for vetoing rules and for threshold budgets, and the
  // frame the main labelled sample is drawn from.
  const rand = seededRandom(seed);
  const byGroup = new Map<string, Prompt[]>();
  for (const p of trainFile) (byGroup.get(p.group) ?? byGroup.set(p.group, []).get(p.group)!).push(p);
  const groups = [...byGroup.values()];
  for (let i = groups.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [groups[i], groups[j]] = [groups[j], groups[i]]; }
  const take = (n: number) => groups.splice(0, n).flat();
  const hist = take(1500), vetoSet = take(500), budgetSet = take(500), frameSet = groups.flat();
  const traffic = (p: Prompt, backgroundUse: 'veto' | 'budget'): ExampleRecord & RecordLike =>
    ({ id: p.id, text: p.text, group: p.group, role: 'background', backgroundUse, source: { kind: 'traffic' }, labels: {} });
  const veto = vetoSet.map((p) => traffic(p, 'veto')), budget = budgetSet.map((p) => traffic(p, 'budget'));
  log(`first round ${hist.length} (toxic ${rate(hist, 'toxic')}, jailbreak ${rate(hist, 'jailbreak')}), veto ${veto.length}, budget ${budget.length}, frame ${frameSet.length}`);

  const mineSet = hist.filter((_, i) => i % 10 < 7), validateSet = hist.filter((_, i) => i % 10 >= 7);
  const mined = mineRules(
    { texts: mineSet.map((p) => p.text), y: mineSet.map((p) => p.truth.toxic), groups: mineSet.map((p) => p.group), backgroundRecords: veto },
    { minGroups: 3, maxBackgroundRate: 0.01, maxRules: 25 },
  );
  const { kept } = validateRules(mined, validateSet.map((p) => p.text), validateSet.map((p) => p.truth.toxic), { maxFalsePositives: 1 });
  log(`mined ${mined.length} toxic rules, ${kept.length} survive validation`);
  const candidateMatcher = ruleSetMatcher(buildRuleSet('tc-candidates', [{ label: 'toxic', rules: kept }], { createdAt: '2026-10-03T00:00:00Z' }));

  // ---------------------------------------------------------------------------------------------
  section('2 sample');
  const round0 = fitLogistic(hist.map(emb), hist.map((p) => p.truth.toxic), { C: 1, classWeight: 'balanced' });
  const stratScore = sampleOptions.stratify?.fit(hist) ?? ((p: Prompt) => sigmoid(decisionFunction(round0, emb(p))));
  const frame: FrameItem[] = frameSet.map((p) => ({ id: p.id, text: p.text, group: p.group, signals: { ruleFires: candidateMatcher.match(p.text) !== null, score: stratScore(p), slice: lengthBucket(p.text) } }));
  const designOptions = { frame, scoreBands: [0.9, 0.5], scoringModel: sampleOptions.stratify?.name ?? 'round0-logistic-mpnet', mergeSmallStrata: true, roleSplit: { train: 0.2, calibration: 0.4, test: 0.4 } } as const;
  const requested = Math.min(frame.length, sampleOptions.total ?? 1800);
  let main: ReturnType<typeof designSample>;
  let pilotLabels = 0;
  if (sampleOptions.method === 'expected-positives') {
    // Pilot (proportional, labelled): per-stratum positive rates that size the main sample.
    const pilot = designSample({ ...designOptions, allocation: { total: 300, method: 'proportional' }, seed: 101 + offset });
    pilotLabels = pilot.records.length;
    const priors = Object.fromEntries(TC_HEADS.map((h) => [h, {} as Record<string, { positives: number; labelled: number }>]));
    for (const r of pilot.records) {
      const st = (r.source as { stratum: string }).stratum;
      for (const h of TC_HEADS) { priors[h][st] ??= { positives: 0, labelled: 0 }; priors[h][st].labelled++; priors[h][st].positives += byId.get(r.id)!.truth[h]; }
    }
    const sizing = { priors, minExpectedPositives: 6 };
    const total = Math.min(frame.length, Math.max(requiredSampleSize({ ...designOptions, allocation: sizing }), requested));
    main = designSample({ ...designOptions, allocation: { total, method: 'expected-positives', ...sizing }, seed: 202 + offset });
  } else if (sampleOptions.topFactor && sampleOptions.topFactor !== 1) {
    main = oversampledTop(designOptions, requested, sampleOptions.topFactor, 202 + offset);
  } else {
    main = designSample({ ...designOptions, allocation: { total: requested, method: 'proportional' }, seed: 202 + offset });
  }
  log(`main design ${main.designId}: ${main.records.length} of ${frame.length} frame prompts in ${main.design.strata.length} strata`);
  for (const w of main.design.warnings) log(`  warning: ${w}`);

  // ---------------------------------------------------------------------------------------------
  section('3 label (blinded queue, simulated reviewer)');
  const queue = labelQueue({ items: main.records.map((record) => ({ record, mechanism: 'sampled' as const, heads: [...TC_HEADS] })), budget: { sampled: main.records.length, retrieved: 0, verification: 0 }, seed: 303 + offset });
  // The simulated reviewer answers with the dataset's labels and skips 2% of items at random.
  const reviewer = seededRandom(404 + offset);
  const reviews = queue.review.filter(() => reviewer() >= 0.02).map((item) => {
    const p = byId.get(queue.key.items[item.itemId].record.id)!;
    return { itemId: item.itemId, labels: Object.fromEntries(item.heads.map((h) => [h, p.truth[h as TcHead]])) as Record<string, 0 | 1> };
  });
  const labelled = applyReviews(queue.key, reviews);
  const calRecords = labelled.records.filter((r) => r.role === 'calibration');
  for (const h of TC_HEADS) log(`calibration positives, ${h}: ${calRecords.filter((r) => r.labels[h] === 1).length} of ${calRecords.length}`);
  const framed = frameSet.filter((p) => main.stratumOf[p.id] !== undefined);
  const faCert = certifyFalseAlarms(hitMatrix(kept, calRecords.map((r) => r.text)), {
    y: calRecords.map((r) => r.labels.toxic as 0 | 1), maxRate: 0.01, design: designOf(calRecords),
    frame: { strata: framed.map((p) => main.stratumOf[p.id]), fired: hitMatrix(kept, framed.map((p) => p.text)) },
  });
  const ruleSet = buildRuleSet('2026-10-03-toxicchat', [{ label: 'toxic', rules: kept.slice(0, faCert.certified) }], { createdAt: '2026-10-03T00:00:00Z' });
  const matcher = ruleSetMatcher(ruleSet);
  log(`rules: ${faCert.certified} of ${kept.length} certified to fire on <= 1% of negatives (${faCert.method}, 95%)`);
  for (const r of ruleSet.rules.slice(0, 8)) log(`  ${r.id.padEnd(10)} ${r.regex.join(' & ')}`);

  const records: ExampleRecord[] = [
    ...hist.map((p): ExampleRecord => ({ id: p.id, text: p.text, group: p.group, role: 'train', source: { kind: 'traffic' }, labels: { ...p.truth }, labelledBy: 'human' })),
    ...labelled.records,
  ];
  const split = records.map((r) => r.role as Split);
  const label = (h: TcHead) => records.map((r) => (r.labels[h] === 0 || r.labels[h] === 1 ? r.labels[h] : null));

  /** One trainHeads run on this round's records. `C` applies to every head given. */
  const train = (policy: (h: TcHead) => HeadPolicy, options: { heads?: readonly TcHead[]; C?: number; log?: (line: string) => void } = {}) => trainHeads({
    X: records.map(emb), split, records, designs: [main.design], seed: 1, calibrationAlpha: 0.01, C: options.C ?? 1,
    // Budget traffic holds the positives too (toxic ~7.5%), so the budgets sit well above prevalence.
    background: { X: budget.map(emb), records: budget, maxRate: { toxic: 0.2, jailbreak: 0.08 } },
    heads: (options.heads ?? TC_HEADS).map((h) => ({ name: h, y: label(h), policy: policy(h), type: 'linear' as const, ...(h === 'toxic' ? { baseline: records.map((r) => matcher.match(r.text) !== null) } : {}) })),
    log: options.log ?? (() => {}),
  });

  const labelsSpent = main.records.length + pilotLabels;
  return { hist, veto, budget, budgetSet, frameSet, main, kept, faCert, ruleSet, matcher, labelled, records, split, label, train, labelsSpent };
}

export type ToxicRound = ReturnType<typeof buildToxicRound>;
