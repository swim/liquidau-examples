/**
 * The release pipeline end to end on a benchmark, as an acceptance test: plan -> a simulated
 * reviewer answers the queue from the true labels (skipping 2%) -> train -> load the release as a
 * runtime would -> check every promise on the fully labelled future file, through the two tiers
 * (rules first, the model for what they don't settle).
 *
 * Usage: release-simulate.ts <abcd|ade|toxicchat> [seed]   Writes out/release-sim/<dataset>/.
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { scoreEmbedding, settleWithRules } from '@liquidau/embedding-classifier';
import { ruleSetMatcher } from '@liquidau/rule-miner';
import { seededRandom } from '@liquidau/solvers';

import { loadAbcd, loadAde, type Doc } from './benchmarks.ts';
import { makeLayerEmbedder } from './embed-layers.ts';
import { loadRelease, plan, train, type LabelledMessage, type Message, type ReleaseConfig } from './release.ts';
import { ClefTeacher } from './teacher/clef.ts';
import { TOXICCHAT_QUESTIONS } from './teacher/questions.ts';
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
const out = join(ROOT, 'out', 'release-sim', name);
const config: ReleaseConfig = { ...cfg, version: `${name}-${seed}`, seed, out, log: say };

const { queue } = await plan(config, traffic, history);
// The simulated reviewer reads each item's text (item ids are opaque) and answers from the true labels,
// skipping 2% of items as real reviewers do.
const byText = new Map(trainDocs.map((d) => [d.text, d.truth]));
const skipper = seededRandom(404 + seed);
const finalReviews = queue.filter(() => skipper() >= 0.02).map((item) => ({ itemId: item.itemId, labels: Object.fromEntries(item.heads.map((h) => [h, byText.get(item.text)![h]])) as Record<string, 0 | 1> }));
await train(config, traffic, history, finalReviews);

// Acceptance: load the release as a runtime would, then the two tiers on the future file.
const embedder = await makeLayerEmbedder(join(ROOT, '.cache', 'onnx', 'mpnet-layers.onnx'), { layers: [4, 8, 12] });
const runtimeSpec = embedder.spec;
const rel = loadRelease(join(out, 'release'), 'shadow', runtimeSpec);
const matcher = ruleSetMatcher(rel.ruleSet);
const heads = cfg.heads.map((h) => h.name);
let settledCount = 0;
const fired: Record<string, boolean[]> = Object.fromEntries(heads.map((h) => [h, []]));
for (let i = 0; i < future.length; i += 32) {
  const batch = future.slice(i, i + 32);
  const vectors = await embedder.embed(batch.map((d) => d.text));
  batch.forEach((d, k) => {
    const ev = matcher.evaluate(d.text);
    const s = settleWithRules(cfg.policy, ev);
    if (s.settled) settledCount++;
    const dismissed = ev.dismissed.filter((h) => heads.includes(h));
    const scores = scoreEmbedding(rel.artifact, vectors[k]);
    for (const h of heads) {
      // Each head on its own (its promise is per head): a rule hit fires it, dismissal clears it, else its threshold.
      const ruleHit = ev.fired?.label === h;
      fired[h].push(ruleHit || (!dismissed.includes(h) && scores[h]! >= rel.artifact.heads[h]!.threshold));
    }
  });
}
const lines = [`# Release acceptance: ${cfg.name} (seed ${seed})`, '', `Future file: ${future.length} messages, fully labelled. Release loaded as a runtime would (validated, rule set paired, embedder spec checked). Gates passed: ${rel.manifest.artifact.gates_passed}; labels spent: ${rel.manifest.labels_spent}.`, '',
  '| head | type | promise | met on the future? | recall | false alarms | precision | fires on |', '|---|---|---|---|---|---|---|---|'];
for (const h of cfg.heads) {
  let tp = 0, fp = 0, pos = 0;
  future.forEach((d, i) => { const y = d.truth[h.name]; pos += y; if (fired[h.name][i]) { if (y) tp++; else fp++; } });
  const recall = tp / pos, fa = fp / (future.length - pos), precision = tp + fp ? tp / (tp + fp) : NaN;
  const g = rel.artifact.heads[h.name]!.guarantee;
  const promised = g && g.kind !== 'none';
  const met = !promised ? '-' : h.kind === 'recall' ? (recall >= h.target ? 'yes' : 'NO') : (precision >= h.target ? 'yes' : 'NO');
  const fireRate = fired[h.name].filter(Boolean).length / future.length;
  lines.push(`| ${h.name} | ${rel.manifest.heads[h.name].type} | ${promised ? `${h.kind} ≥ ${h.target} (${g!.kind})` : 'none (gates refused)'} | ${met} | ${recall.toFixed(3)} | ${(100 * fa).toFixed(2)}% | ${precision.toFixed(3)} | ${(100 * fireRate).toFixed(1)}%${h.maxFireRate !== undefined ? ` (cap ${100 * h.maxFireRate}%)` : ''} |`);
}
lines.push('', `Rules tier: settles ${(100 * settledCount / future.length).toFixed(1)}% of the future file (frame estimate at release: ${(100 * rel.manifest.rules_tier_share_of_frame).toFixed(1)}%).`);
writeFileSync(join(out, 'acceptance.md'), lines.join('\n'));
console.log(lines.join('\n'));
