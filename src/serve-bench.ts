/**
 * Function-shaped serving measurements for a split deployment, each function in a FRESH process (a
 * cold start): module load, model load (model function only), release load and validation, the first
 * request, resident memory, then steady-state latency - the rules function over every message, the
 * model function over requests the rules can't settle, embedding each one for real (no cache).
 *
 * This machine is not a function's CPU (a Lambda's vCPU share scales with its configured memory), so
 * read the numbers as a guide to proportions, and measure on the target platform before sizing.
 * Usage: serve-bench.ts            all releases and both functions, writes out/serve-bench.md
 *        serve-bench.ts <case> <rules|model> [requests]   one measurement, printed as JSON
 */
const started = performance.now();
const { spawnSync } = await import('node:child_process');
const { existsSync, readFileSync, writeFileSync } = await import('node:fs');
const { join } = await import('node:path');

const ROOT = join(import.meta.dirname, '..');
const CASES: Record<string, { dir: string }> = {
  banking77: { dir: join(ROOT, 'out', 'release') },
  abcd: { dir: join(ROOT, 'out', 'release-sim', 'abcd', 'release') },
  ade: { dir: join(ROOT, 'out', 'release-sim', 'ade', 'release') },
  toxicchat: { dir: join(ROOT, 'out', 'release-sim', 'toxicchat', 'release') },
};
const [caseArg, tierArg, requestsArg] = process.argv.slice(2);

if (!caseArg) {
  const lines = ['# Serving measurements (function-shaped, this machine)', '', 'Each function measured in a fresh process. Model function latency embeds every request (no cache).', '',
    '| release | function | import | model load | release load | first request | resident memory after loading | requests | median | p95 |', '|---|---|---|---|---|---|---|---|---|---|'];
  for (const [name, c] of Object.entries(CASES)) {
    if (!existsSync(join(c.dir, 'deploy.json'))) continue;
    for (const tier of ['rules', 'model']) {
      const run = spawnSync(process.execPath, [...process.execArgv, import.meta.filename, name, tier, '200'], { encoding: 'utf8', maxBuffer: 1 << 26 });
      const out = run.stdout.trim().split('\n').pop()!;
      let r: Record<string, number>;
      try { r = JSON.parse(out); } catch { console.error(`${name} ${tier} failed:\n${run.stderr.slice(-2000)}`); process.exitCode = 1; continue; }
      const ms = (v: number) => (v >= 100 ? `${(v / 1000).toFixed(2)} s` : v >= 1 ? `${v.toFixed(1)} ms` : `${(1000 * v).toFixed(0)} µs`);
      const row = `| ${name} | ${tier} | ${ms(r.importMs)} | ${tier === 'model' ? ms(r.modelMs) : '-'} | ${ms(r.loadMs)} | ${ms(r.firstMs)} | ${(r.rssBytes / 2 ** 20).toFixed(0)} MB | ${r.requests} | ${ms(r.p50Ms)} | ${ms(r.p95Ms)} |`;
      console.log(row);
      lines.push(row);
    }
  }
  writeFileSync(join(ROOT, 'out', 'serve-bench.md'), lines.join('\n') + '\n');
} else {
  const c = CASES[caseArg];
  // Only what each function needs: the rules function imports the router and a release source alone.
  const { loadRulesTier, loadRouter } = await import('@liquidau/router');
  const { fileSource } = await import('./release-source.ts');
  const importMs = performance.now() - started;
  /** The benchmark's messages, loaded AFTER the function's memory is measured. */
  const loadTexts = async () => {
    if (caseArg === 'banking77') return (await (await import('./data.ts')).loadBanking77()).map((m) => m.text);
    const b = await import('./benchmarks.ts');
    if (caseArg === 'abcd') return b.loadAbcd().future.map((d) => d.text);
    if (caseArg === 'ade') return b.loadAde().future.map((d) => d.text);
    return (await (await import('./toxicchat.ts')).loadToxicChat()).filter((p) => p.split === 'test').map((p) => p.text);
  };
  let rssBytes = 0;
  let texts: string[] = [];
  const deploy = JSON.parse(readFileSync(join(c.dir, 'deploy.json'), 'utf8')) as { manifestKey: string; manifestSha256: string };
  const { staticEncoder } = await import('./static-host.ts');
  const pin = { source: fileSource(c.dir), manifestKey: deploy.manifestKey, expectedManifestSha256: deploy.manifestSha256, mode: 'shadow' as const, timeoutMs: 60_000, monitoring: { sampleRate: 0, samplingSalt: 'bench' }, staticEncoder };
  const quantile = (xs: number[], q: number) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
  let modelMs = 0, loadMs: number, firstMs: number;
  const unsettled: string[] = [];
  const lat: number[] = [];
  if (tierArg === 'rules') {
    const t = performance.now();
    const tier = await loadRulesTier(pin);
    loadMs = performance.now() - t;
    rssBytes = process.memoryUsage().rss;
    texts = await loadTexts();
    const f = performance.now();
    await tier.handle({ text: texts[0], requestId: 'first' });
    firstMs = performance.now() - f;
    for (const [i, text] of texts.entries()) { const s = performance.now(); await tier.handle({ text, requestId: `r${i}` }); lat.push(performance.now() - s); }
  } else {
    const tm = performance.now();
    // The encoder the release's classifier was trained with, embedding every request for real (no cache).
    const { gemmaEncoder, layerEncoder, transformersEncoder } = await import('./router-host.ts');
    const spec = (JSON.parse(readFileSync(join(c.dir, 'classifier.json'), 'utf8')) as { embedding: import('@liquidau/embedding-classifier').EmbeddingSpec }).embedding;
    const { GEMMA_MODEL, makeGemmaEmbedder } = await import('./embed-gemma.ts');
    let encoder;
    if (spec.model_id === GEMMA_MODEL) {
      encoder = await gemmaEncoder(await makeGemmaEmbedder({ maxChars: spec.max_chars }));
    } else if (spec.layers) {
      const { makeLayerEmbedder } = await import('./embed-layers.ts');
      const modelPath = join(ROOT, '.cache', 'onnx', 'mpnet-layers.onnx');
      encoder = await layerEncoder(await makeLayerEmbedder(modelPath, { layers: spec.layers }), modelPath);
    } else {
      const { pipeline } = await import('@huggingface/transformers');
      const extract = await pipeline('feature-extraction', spec.model_id, { dtype: 'q8' });
      encoder = await transformersEncoder(spec.model_id, spec.dimensions, async (t) => (await extract([...t], { pooling: 'mean', normalize: true })).tolist() as number[][]);
    }
    modelMs = performance.now() - tm;
    const t = performance.now();
    const router = await loadRouter({ ...pin, encoder });
    loadMs = performance.now() - t;
    rssBytes = process.memoryUsage().rss;
    // Which messages reach the model function: the rules tier's own answer (a separate function in production).
    texts = await loadTexts();
    const rulesOnly = await loadRulesTier(pin);
    for (const [i, text] of texts.entries()) if (!(await rulesOnly.handle({ text, requestId: `p${i}` })).answered) unsettled.push(text);
    const sample = unsettled.slice(0, Number(requestsArg ?? 200));
    const forward = (text: string, id: string) => ({ schema: 'liquidau-router-forward/1' as const, manifestSha256: router.manifestSha256, requestId: id, text, purpose: 'decide' as const, inclusionProbability: 1 });
    const f = performance.now();
    await router.routeForwarded(forward(sample[0] ?? texts[0], 'first'));
    firstMs = performance.now() - f;
    for (const [i, text] of sample.entries()) { const s = performance.now(); await router.routeForwarded(forward(text, `m${i}`)); lat.push(performance.now() - s); }
  }
  console.log(JSON.stringify({ importMs, modelMs, loadMs, firstMs, rssBytes, requests: lat.length, p50Ms: quantile(lat, 0.5), p95Ms: quantile(lat, 0.95), unsettled: unsettled.length, messages: texts.length }));
}
