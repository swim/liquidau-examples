/**
 * Checks src/serve.ts against the full path on every Banking77 message, with the shipped artifact and
 * rule set (run `npm start` first): skipping the embedding must never change a decision. Also runs a
 * policy where the rules' head can be suppressed, where nothing may be skipped.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { decide, scoreEmbedding, type DecisionPolicy } from '@liquidau/embedding-classifier';
import { ruleSetMatcher } from '@liquidau/rule-miner';

import { loadBanking77 } from './data.ts';
import { makeEmbedder } from './embed.ts';
import { createHandler } from './serve.ts';

const OUT = join(import.meta.dirname, '..', 'out');
const artifact = JSON.parse(readFileSync(join(OUT, 'artifact.json'), 'utf8'));
const ruleSet = JSON.parse(readFileSync(join(OUT, 'ruleset.json'), 'utf8'));
const messages = await loadBanking77();
const embedder = await makeEmbedder();
await embedder.embedMany(messages.map((m) => m.text)); // all cached: the check measures the handler, not the model
let embedCalls = 0;
const embed = async (text: string) => { embedCalls++; return (await embedder.embedMany([text]))[0]; };
const matcher = ruleSetMatcher(ruleSet);

/** The reference: always embed, rules OR classifier, then the policy. */
const reference = async (text: string, policy: DecisionPolicy<string>) => {
  const scores = scoreEmbedding(artifact, await embed(text));
  const hit = matcher.match(text);
  if (hit && artifact.heads[hit.label]) scores[hit.label] = Math.max(scores[hit.label] ?? 0, artifact.heads[hit.label].threshold);
  return decide(artifact, scores, policy);
};

let failed = false;
const policies: Array<[string, DecisionPolicy<string>, boolean]> = [
  ['urgent first (skipping allowed)', { priority: ['urgent', 'refund'] }, true],
  ['urgent suppressible (no skipping)', { priority: ['urgent', 'refund'], suppress: [{ when: ['refund'], heads: ['urgent'] }] }, false],
  ['refund first (no skipping)', { priority: ['refund', 'urgent'] }, false],
];
for (const [name, policy, mayskip] of policies) {
  const handle = createHandler({ artifact, ruleSet, policy, embed, monitorShare: 0 });
  let mismatches = 0, skipped = 0, ruleDecided = 0;
  let handlerMs = 0;
  for (const m of messages) {
    const before = embedCalls, t = performance.now();
    const served = await handle(m.text);
    handlerMs += performance.now() - t;
    const handlerEmbedded = embedCalls > before;
    const ref = await reference(m.text, policy);
    if (served.head !== ref.head || served.reason !== ref.reason) mismatches++;
    if (!served.embedded) skipped++;
    if (served.rule) ruleDecided++;
    if (served.embedded !== handlerEmbedded) mismatches++;
  }
  const ok = mismatches === 0 && (mayskip ? skipped > 0 : skipped === 0);
  failed ||= !ok;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}: ${messages.length} messages, ${mismatches} mismatches with the full path, rules fired on ${ruleDecided}, embedding skipped for ${skipped} (${(100 * skipped / messages.length).toFixed(1)}%); handler ${(1000 * handlerMs / messages.length).toFixed(0)} µs per message with cached embeddings`);
}

// Monitoring share: with monitorShare = 1 nothing is skipped; with 0.05 about 5% of rule-decided messages are embedded.
const sampled = createHandler({ artifact, ruleSet, policy: { priority: ['urgent', 'refund'] }, embed, monitorShare: 0.05, sample: (() => { let i = 0; return () => ((i++ * 0.6180339887) % 1); })() });
let ruleHits = 0, embeddedHits = 0;
for (const m of messages) { const s = await sampled(m.text); if (s.rule) { ruleHits++; if (s.embedded) embeddedHits++; } }
console.log(`monitorShare 0.05: ${embeddedHits} of ${ruleHits} rule-decided messages embedded for monitoring (${(100 * embeddedHits / ruleHits).toFixed(1)}%)`);
if (failed) process.exitCode = 1;
