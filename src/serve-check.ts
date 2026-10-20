/**
 * Checks the router-based handler (src/serve.ts) against the always-embed reference on every Banking77
 * message (run `npm start` first, which writes the release). For each policy a release is built with
 * that policy and evaluation evidence from the same held-out test records, then served in shadow mode:
 * every decision must equal the reference evaluator's on the same rules findings and always-embed
 * scores, and the handler must embed exactly when the router reports it did. A policy where a rule's
 * head can be outranked or suppressed must settle nothing on the rules alone.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { scoreEmbedding, type ClassifierArtifact } from '@liquidau/embedding-classifier';
import { createDecisionEvaluator, memorySource, REVIEW_ADAPTER_SCHEMA, sha256Hex, type ReviewOptions, type RouterEvent } from '@liquidau/router';
import { ruleSetMatcher, type RuleSet } from '@liquidau/rule-miner';

import { loadBanking77 } from './data.ts';
import { makeEmbedder, walkthroughEncoder } from './embed.ts';
import { packageVersions } from './router-host.ts';
import { buildRouterRelease, policyFor } from './router-release.ts';
import { createHandler } from './serve.ts';

const OUT = join(import.meta.dirname, '..', 'out');
const RELEASE = join(OUT, 'release');
const artifact = JSON.parse(readFileSync(join(RELEASE, 'classifier.json'), 'utf8')) as ClassifierArtifact;
const ruleSet = JSON.parse(readFileSync(join(RELEASE, 'rules.json'), 'utf8')) as RuleSet;
const heldOut = JSON.parse(readFileSync(join(OUT, 'heldout.json'), 'utf8')) as Array<{ id: string; labels: Record<string, 0 | 1 | null> }>;
const messages = await loadBanking77();
const byId = new Map(messages.map((m) => [m.id, m]));
const embedder = await makeEmbedder();
const vectors = await embedder.embedMany(messages.map((m) => m.text)); // all cached: the check measures the handler, not the model
const vectorOf = new Map(messages.map((m, i) => [m.text, vectors[i]]));
let embedCalls = 0;
const encoder = await walkthroughEncoder(async (texts) => { embedCalls += texts.length; return embedder.embedMany([...texts]); });
const matcher = ruleSetMatcher(ruleSet);
const heads = Object.keys(artifact.heads);
const model = { heads: Object.fromEntries(heads.map((h) => [h, { threshold: artifact.heads[h]!.threshold, reviewFloor: artifact.heads[h]!.review_floor }])) };

let failed = false;
const policies: Array<[string, ReturnType<typeof policyFor>, boolean]> = [
  ['urgent first (rules may settle)', policyFor(heads, ['urgent', 'refund']), true],
  ['urgent suppressible (nothing settles)', policyFor(heads, ['urgent', 'refund'], [{ when: ['refund'], heads: ['urgent'] }]), false],
  ['refund first (nothing settles)', policyFor(heads, ['refund', 'urgent']), false],
];
// A stub review adapter (answers the first offered route): the decision must be unchanged by design.
const digest = async (t: string) => sha256Hex(new TextEncoder().encode(t));
const stubReview: ReviewOptions = {
  timeoutMs: 1_000, humanSampleRate: 0.05,
  adapter: {
    identity: { schema: REVIEW_ADAPTER_SCHEMA, provider: 'stub', model: 'first-route', promptSha256: await digest('serve-check'), outputSchemaSha256: await digest('{"routeId":"string|null","confidence":"number|null"}') },
    async review(input) { return { routeId: input.routes[0]?.id ?? null, confidence: 0.5 }; },
  },
};
for (const withAdapter of [false, true])
for (const [name, policy, maySettle] of policies) {
  const { built } = await buildRouterRelease({
    releaseId: `serve-check-${name.split(' ')[0]}-${name.split(' ')[1]}`, createdAt: artifact.created_at, artifact, ruleSet, policy,
    items: heldOut.map((h) => ({ id: h.id, text: byId.get(h.id)!.text, x: vectorOf.get(byId.get(h.id)!.text)!, labels: h.labels })),
    dataset: { id: 'banking77-test', split: 'test' }, criteria: { heads: { urgent: { minRecall: 0.85 }, refund: { minPrecision: 0.7 } }, compare: 'point' },
    embedding: encoder.identity, packages: packageVersions(),
  });
  const handler = await createHandler({ source: memorySource(built.files), manifestKey: built.manifestKey, manifestSha256: built.manifestSha256, mode: 'shadow', encoder, monitoringRate: 0, ...(withAdapter ? { review: stubReview } : {}) });
  const evaluator = createDecisionEvaluator(model, policy);
  let mismatches = 0, settled = 0, ruleFired = 0, handlerMs = 0, reviews = 0, answered = 0;
  for (const m of messages) {
    const before = embedCalls, t = performance.now();
    const r = await handler.handle(m.text, m.id);
    handlerMs += performance.now() - t;
    if (r.outcome !== 'shadow') { mismatches++; continue; }
    const findings = matcher.evaluate(m.text);
    const ref = evaluator.decide(findings, scoreEmbedding(artifact, vectorOf.get(m.text)!) as Record<string, number>);
    const { answer, answerError, forHuman, ...decision } = r.candidate as typeof r.candidate & { answer?: unknown; answerError?: unknown; forHuman?: boolean };
    if (JSON.stringify(ref) !== JSON.stringify(decision)) mismatches++;
    if (decision.outcome === 'review') { reviews++; if (answer) answered++; }
    if (!withAdapter && (answer || answerError || forHuman !== undefined)) mismatches++;
    if (withAdapter && decision.outcome === 'review' && (!answer || answerError)) mismatches++;
    if (r.embedded !== (embedCalls > before)) mismatches++;
    if (!r.embedded) settled++;
    if (findings.fired) ruleFired++;
  }
  await handler.close();
  const ok = mismatches === 0 && (maySettle ? settled > 0 : settled === 0) && (!withAdapter || answered === reviews);
  failed ||= !ok;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${withAdapter ? ` + review adapter (${answered}/${reviews} reviews answered)` : ''}: ${messages.length} messages, ${mismatches} mismatches with the always-embed reference, rules fired on ${ruleFired}, settled without embedding ${settled} (${(100 * settled / messages.length).toFixed(1)}%); handler ${(1000 * handlerMs / messages.length).toFixed(0)} µs per message with cached embeddings`);
}

// Monitoring: a deterministic 5% of rules-settled requests are still embedded (after the result).
const { built } = await buildRouterRelease({
  releaseId: 'serve-check-monitoring', createdAt: artifact.created_at, artifact, ruleSet, policy: policyFor(heads, ['urgent', 'refund']),
  items: heldOut.map((h) => ({ id: h.id, text: byId.get(h.id)!.text, x: vectorOf.get(byId.get(h.id)!.text)!, labels: h.labels })),
  dataset: { id: 'banking77-test', split: 'test' }, criteria: { heads: { urgent: { minRecall: 0.85 } }, compare: 'point' }, embedding: encoder.identity, packages: packageVersions(),
});
const events: RouterEvent[] = [];
const monitored = await createHandler({ source: memorySource(built.files), manifestKey: built.manifestKey, manifestSha256: built.manifestSha256, mode: 'shadow', encoder, monitoringRate: 0.05, observer: (e) => { events.push(e); } });
for (const m of messages) await monitored.handle(m.text, m.id);
await monitored.close();
const settledEvents = events.filter((e) => e.type === 'decision' && e.rulesSettled);
const samples = events.filter((e) => e.type === 'diagnostic').length;
console.log(`monitoring 0.05: ${samples} of ${settledEvents.length} rules-settled messages embedded for monitoring (${(100 * samples / settledEvents.length).toFixed(1)}%), none of them delaying a result`);
if (failed) process.exitCode = 1;
