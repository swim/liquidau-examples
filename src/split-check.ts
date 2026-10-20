/**
 * Checks the split deployment (src/hosts.ts: a rules function forwarding JSON to a model function)
 * against one router, on every message of each release: the walkthrough's Banking77 release (run
 * `npm start`) and the release simulations' (run `npm run release:simulate -- <dataset>`). Results must
 * be identical and the two functions' telemetry together must equal the single router's. The first
 * 200 messages also go through the Lambda wrappers as function-URL events. Embeddings come from the
 * caches, so this measures the routing, not the model.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { loadRouter, RouterLoadError, type RouterEvent, type RouterMode } from '@liquidau/router';

import { createModelFunction, createRulesFunction, lambdaHandler, type ReleasePin } from './hosts.ts';
import { releaseCases, ROOT } from './release-cases.ts';
import { fileSource } from './router-host.ts';
import { staticEncoder } from './static-host.ts';

const STATIC = staticEncoder;

const MONITORING = { sampleRate: 0.05, samplingSalt: 'split-check' };
const comparable = (events: RouterEvent[]) => events.map((e) => { const { timings: _t, ...rest } = e as RouterEvent & { timings?: unknown }; return JSON.stringify(rest); }).sort();

const cases = await releaseCases();
if (!cases.length) throw new Error('no releases found: run npm start and/or npm run release:simulate first');

const lines = ['# Split deployment check', '', 'Rules function -> JSON -> model function (src/hosts.ts), against one router, on every message. Embeddings cached.', '',
  '| release | mode | messages | answered by the rules function | forwarded to the model function | monitoring forwards | result mismatches | telemetry identical |', '|---|---|---|---|---|---|---|---|'];
let failed = false;
for (const c of cases) {
  const deploy = JSON.parse(readFileSync(join(c.dir, 'deploy.json'), 'utf8')) as { manifestKey: string; manifestSha256: string };
  const encoder = await c.encoder();
  const single: RouterEvent[] = [], split: RouterEvent[] = [];
  const pinFor = (mode: RouterMode, sink: RouterEvent[]): ReleasePin => ({ source: fileSource(c.dir), manifestKey: deploy.manifestKey, manifestSha256: deploy.manifestSha256, mode, monitoring: MONITORING, observer: (e) => { sink.push(e); } });
  let mode: RouterMode = 'enforce';
  const loadSingle = (m: RouterMode) => loadRouter({ source: fileSource(c.dir), manifestKey: deploy.manifestKey, expectedManifestSha256: deploy.manifestSha256, mode: m, encoder, timeoutMs: 60_000, monitoring: MONITORING, staticEncoder: STATIC, observer: (e) => { single.push(e); } });
  let router;
  try { router = await loadSingle('enforce'); } catch (e) { if (!(e instanceof RouterLoadError && e.code === 'GATES_FAILED')) throw e; mode = 'shadow'; router = await loadSingle('shadow'); }
  const model = await createModelFunction(pinFor(mode, split), encoder);
  let forwarded = 0, monitors = 0;
  // In-process "network": the forward and the answer cross as JSON.
  const rules = await createRulesFunction(pinFor(mode, split), async (f) => {
    if (f.purpose === 'decide') forwarded++; else monitors++;
    const out = await model.handle(JSON.parse(JSON.stringify(f)));
    await out.after;
    return JSON.parse(JSON.stringify(out.body));
  });
  const lambda = lambdaHandler(async () => rules);
  let mismatches = 0;
  for (const [i, text] of c.texts.entries()) {
    const requestId = `m${i}`;
    const expected = await router.route({ text, requestId });
    let actual: unknown;
    if (i < 200) {
      const res = await lambda({ body: JSON.stringify({ text, requestId }) });
      actual = JSON.parse(res.body);
    } else {
      const out = await rules.handle({ text, requestId });
      await out.after;
      actual = JSON.parse(JSON.stringify(out.body));
    }
    if (JSON.stringify(actual) !== JSON.stringify(expected)) mismatches++;
  }
  await Promise.all([router.flush(), model.router.flush(), rules.tier.flush()]);
  const sameTelemetry = JSON.stringify(comparable(split)) === JSON.stringify(comparable(single));
  failed ||= mismatches > 0 || !sameTelemetry;
  const answered = c.texts.length - forwarded;
  const row = `| ${c.name} | ${mode} | ${c.texts.length} | ${answered} (${(100 * answered / c.texts.length).toFixed(1)}%) | ${forwarded} | ${monitors} | ${mismatches} | ${sameTelemetry ? 'yes' : 'NO'} |`;
  console.log(row);
  lines.push(row);
}
writeFileSync(join(ROOT, 'out', 'split-check.md'), lines.join('\n') + '\n');
if (failed) process.exitCode = 1;
