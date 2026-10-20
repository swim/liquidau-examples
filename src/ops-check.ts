/**
 * Operations on the ABCD release (run `npm run release:simulate -- abcd` first): the release's own
 * sampling frame, routed by the release, is the reference window; live windows are checked against it
 * with src/ops-monitor.ts. Live windows: the two halves of the future file (a later period), the same
 * period with account-access requests surging, and a window where the encoder fails one call in five.
 * Each window also gets each head's prevalence, estimated from its routing rate (BBSE), next to the truth
 * (known here because the simulation's files are fully labelled).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import type { ClassifierArtifact } from '@liquidau/embedding-classifier';
import { EncoderError, loadRouter, type Encoder, type RouterEvent } from '@liquidau/router';
import type { RuleSet } from '@liquidau/rule-miner';
import { seededRandom } from '@liquidau/solvers';

import { loadAbcd } from './benchmarks.ts';
import { checkWindow, estimatePrevalence, windowOf, type HeadConfusion, type WindowReport } from './ops-monitor.ts';
import { releaseEmbedderFor } from './release-encoder.ts';
import { fileSource } from './router-host.ts';
import { staticEncoder } from './static-host.ts';

const ROOT = join(import.meta.dirname, '..');
const RUN = join(ROOT, 'out', 'release-sim', 'abcd'), DIR = join(RUN, 'release');
const deploy = JSON.parse(readFileSync(join(DIR, 'deploy.json'), 'utf8')) as { manifestKey: string; manifestSha256: string };
/** Enforce when the release supports it, else shadow (monitoring is the same either way). */
const MODE = ((JSON.parse(readFileSync(join(RUN, 'release', 'summary.json'), 'utf8')) as { enforcement?: string[] }).enforcement?.length ? 'shadow' : 'enforce') as 'shadow' | 'enforce';
const artifact = JSON.parse(readFileSync(join(DIR, 'classifier.json'), 'utf8')) as ClassifierArtifact;
const ruleSet = JSON.parse(readFileSync(join(DIR, 'rules.json'), 'utf8')) as RuleSet;
const labelOf = new Map(ruleSet.rules.map((r) => [r.id, r.label]));
const heads = Object.keys(artifact.heads);
/** Each head's routing on the release's labelled test split, from its evidence. */
type Count = { count: number; of: number };
const evidence = JSON.parse(readFileSync(join(DIR, 'evidence.json'), 'utf8')) as { perHead: { heads: Record<string, { recall: Count; falseAlarmRate: Count }> } };
const confusion: Record<string, HeadConfusion> = Object.fromEntries(heads.map((h) => {
  const e = evidence.perHead.heads[h]!;
  return [h, { truePositives: e.recall.count, positives: e.recall.of, falsePositives: e.falseAlarmRate.count, negatives: e.falseAlarmRate.of }];
}));
const specs = Object.fromEntries(heads.map((h) => [h, { threshold: artifact.heads[h]!.threshold, review_floor: artifact.heads[h]!.review_floor }]));

const abcd = loadAbcd();
const byId = new Map(abcd.train.map((d) => [d.id, d]));
const frame = (JSON.parse(readFileSync(join(RUN, 'plan', 'plan.json'), 'utf8')) as { frame: string[] }).frame.map((id) => byId.get(id)!.text);
const future = abcd.future;

const cache = await releaseEmbedderFor(artifact.embedding);
await cache.embedMany([...frame, ...future.map((d) => d.text)]);
const encoder = await cache.encoder();
/** The same encoder, failing one call in five (a degraded dependency). */
const flaky = (): Encoder => { const rand = seededRandom(5); return { identity: encoder.identity, embed: async (t, o) => { if (rand() < 0.2) throw new EncoderError('ENCODER_FAILURE', 'provider unavailable'); return encoder.embed(t, o); } }; };

/** Routes texts through the release and returns its telemetry as a window. */
async function windowFor(texts: readonly string[], enc: Encoder) {
  const events: RouterEvent[] = [];
  const router = await loadRouter({ source: fileSource(DIR), manifestKey: deploy.manifestKey, expectedManifestSha256: deploy.manifestSha256, mode: MODE, encoder: enc, timeoutMs: 60_000, monitoring: { sampleRate: 0.05, samplingSalt: 'ops' }, staticEncoder, observer: (e) => { events.push(e); } });
  for (const [i, text] of texts.entries()) await router.route({ text, requestId: `w${i}-${text.length}` });
  await router.close();
  return windowOf(events, heads, (id) => labelOf.get(id));
}

const reference = await windowFor(frame, encoder);
const rand = seededRandom(17);
const pick = <T>(xs: readonly T[], n: number) => Array.from({ length: n }, () => xs[Math.floor(rand() * xs.length)]);
const access = future.filter((d) => d.truth.access === 1), other = future.filter((d) => d.truth.access !== 1);
type Doc = (typeof future)[number];
const windows: Array<[string, Doc[], Encoder]> = [
  ['future file, first half', future.slice(0, 1000), encoder],
  ['future file, second half', future.slice(1000), encoder],
  ['account-access surge (35% of requests)', [...pick(access, 350), ...pick(other, 650)], encoder],
  ['encoder failing one call in five', future.slice(0, 1000), flaky()],
];
const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
const prevalenceRows: string[] = [];
const lines = ['# Operations check: ABCD release', '',
  `Reference window: the release's sampling frame (${reference.requests} requests; ${reference.settled} settled by the rules; ${reference.sampled[heads[0]].length} monitoring samples). Family-wise false-alarm rate 1% per window.`, '',
  '| window | requests | settled by rules | alerts | retrain? | investigate? |', '|---|---|---|---|---|---|'];
const details: string[] = [];
for (const [name, docs, enc] of windows) {
  const live = await windowFor(docs.map((d) => d.text), enc);
  for (const [h, e] of Object.entries(estimatePrevalence(live, confusion))) {
    const truth = docs.filter((d) => d.truth[h] === 1).length / docs.length;
    prevalenceRows.push(`| ${name} | ${h} | ${pct(e.routedRate)} | ${pct(e.prevalence)} (${pct(e.lower)}-${pct(e.upper)}) | ${pct(truth)} |`);
  }
  const r: WindowReport = checkWindow(reference, live, specs, { alpha: 0.01 });
  const row = `| ${name} | ${live.requests} | ${(100 * live.settled / live.requests).toFixed(1)}% | ${r.checks.filter((c) => c.alert).map((c) => c.name).join(', ') || 'none'} | ${r.retrain.recommended ? 'yes' : 'no'} | ${r.investigate.length ? 'yes' : 'no'} |`;
  console.log(row);
  lines.push(row);
  details.push(`### ${name}`, '', ...r.checks.filter((c) => c.alert).map((c) => `- ${c.name} (p ${c.pValue.toExponential(1)}): ${c.detail}`), ...(r.skipped.length ? [`- skipped (too few scores): ${r.skipped.join('; ')}`] : []), '');
}
const prior = Object.fromEntries(heads.map((h) => [h, pct(confusion[h]!.positives / (confusion[h]!.positives + confusion[h]!.negatives))]));
const prevalence = ['## Prevalence from routing rates', '',
  `Each head's prevalence estimated from its routing rate alone (BBSE), using the release's routing on its labelled test split (prevalence there: ${heads.map((h) => `${h} ${prior[h]}`).join(', ')}). Truth is known only because this simulation's files are fully labelled. The estimate assumes label shift (the mix of positives changed, not how they look) and is not validated when the messages themselves change. A precision promise weakens when prevalence falls, so an interval well below the release's prevalence calls for re-thresholding precision heads (README).`, '',
  '| window | head | routed | estimated prevalence (95%) | true prevalence |', '|---|---|---|---|---|', ...prevalenceRows];
console.log(prevalence.join('\n'));
writeFileSync(join(ROOT, 'out', 'ops-check.md'), [...lines, '', ...prevalence, '', ...details].join('\n'));
