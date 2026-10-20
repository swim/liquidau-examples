/**
 * The example's showcase page (site/index.html): a diagram of the routing pathways a message can take,
 * with the share of each dataset's messages on each one and real messages that took it, then a card per
 * dataset with its classifier and its numbers. Everything is read from the releases (manifest,
 * classifier, edge, policy, evidence, summary) and from routing each release's labelled check file
 * through @liquidau/router, so the page cannot drift from what was built. Run after `npm start` and
 * `npm run release:simulate -- <dataset>`; embeddings come from the caches.
 *
 * Usage: site.ts [output directory] (default site/).
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { scoreEmbedding } from '@liquidau/embedding-classifier';
import { createEdgeTier, loadRouter, RouterLoadError, type DecisionEvent, type RouterEvent, type RouterMode } from '@liquidau/router';
import { evaluatePerHead } from '@liquidau/router/evaluation';

import { releaseCases, ROOT } from './release-cases.ts';
import { fileSource } from './router-host.ts';
import { staticEncoder } from './static-host.ts';

const OUT = process.argv[2] ?? join(ROOT, 'site');

/** What each dataset is and what its routes mean (the rest comes from the release). */
const INFO: Record<string, { title: string; blurb: string; routes: Record<string, string> }> = {
  banking77: {
    title: 'Banking77', blurb: 'Online-banking customer messages (PolyAI, CC BY 4.0): 13,083 messages over 77 intents, mapped to two routes.',
    routes: { urgent: 'Lost, stolen or compromised cards and phones; payments, withdrawals or direct debits the customer doesn\'t recognise', refund: 'Refund requests, and refunds not showing up' },
  },
  abcd: {
    title: 'ABCD', blurb: 'ASAPP\'s Action-Based Conversations Dataset (MIT): 10,042 customer-service chats. The text is the customer\'s opening turns, before the agent replies.',
    routes: { access: 'Account access: recover a password or username, reset two-factor', refund: 'Initiate, check or update a refund' },
  },
  ade: {
    title: 'ADE', blurb: 'ADE Corpus V2: 23,516 sentences from PubMed case reports. Does the sentence describe an adverse drug event?',
    routes: { ade: 'Describes an adverse drug event' },
  },
  toxicchat: {
    title: 'ToxicChat', blurb: 'LMSYS ToxicChat 0124 (CC BY-NC 4.0): real user prompts to the Vicuna demo, labelled for toxicity and jailbreaking. English only.',
    routes: { toxic: 'Toxic prompt', jailbreak: 'Jailbreak attempt' },
  },
};

/** No release can be built for CFPB: the gates refuse it. Lab figures (liquidau-lab RESEARCH.md, stage EL). */
const REFUSED = [{
  key: 'cfpb', title: 'CFPB complaints', blurb: 'US Consumer Financial Protection Bureau complaint narratives (CC0). A real time split: 2022 narratives to build on, July-December 2023 to check.',
  routes: [
    { id: 'fraud', definition: 'The issue names fraud, a scam, identity theft or unauthorised activity', promise: 'recall ≥ 0.85', lab: 'never guaranteed (0 of 5 seeds, every configuration)' },
    { id: 'debt', definition: 'Product: debt collection', promise: 'precision ≥ 0.8', lab: 'guaranteed in 2 of 5 seeds; the promise never held on the 2023 file' },
  ],
  reason: 'The release gates refuse it: no head keeps its promise across the 2022 → 2023 shift.',
}];

type Pathway = 'rule' | 'edge' | 'settled' | 'model' | 'review' | 'none' | 'unavailable';
const pathwayOf = (e: DecisionEvent): Pathway =>
  e.outcome === 'unavailable' ? 'unavailable'
    : e.outcome === 'route' ? (e.mechanism === 'rule' ? 'rule' : e.mechanism === 'edge' ? 'edge' : 'model')
      : e.rulesSettled ? 'settled' : e.outcome === 'review' ? 'review' : 'none';

const wilson = (k: number, n: number) => {
  if (!n) return null;
  const z = 1.959964, p = k / n, d = 1 + (z * z) / n, mid = (p + (z * z) / (2 * n)) / d, half = (z / d) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return { rate: p, lo: Math.max(0, mid - half), hi: Math.min(1, mid + half), k, n };
};
const read = (dir: string, file: string) => (existsSync(join(dir, file)) ? JSON.parse(readFileSync(join(dir, file), 'utf8')) : null);
const stableHash = (s: string) => { let h = 2166136261; for (let k = 0; k < s.length; k++) h = Math.imul(h ^ s.charCodeAt(k), 16777619); return h >>> 0; };

const releases = [];
for (const c of await releaseCases()) {
  const info = INFO[c.key];
  const deploy = read(c.dir, 'deploy.json') as { manifestKey: string; manifestSha256: string };
  const manifest = read(c.dir, 'manifest.json'), classifier = read(c.dir, 'classifier.json'), evidence = read(c.dir, 'evidence.json'), policy = read(c.dir, 'policy.json');
  const summary = read(c.dir, 'summary.json'), edge = manifest.files.edge ? read(c.dir, manifest.files.edge.key) : null;
  const encoder = await c.encoder();
  const events = new Map<string, DecisionEvent>();
  const load = (mode: RouterMode) => loadRouter({ source: fileSource(c.dir), manifestKey: deploy.manifestKey, expectedManifestSha256: deploy.manifestSha256, mode, encoder, timeoutMs: 60_000, monitoring: { sampleRate: 0, samplingSalt: 'site' }, staticEncoder, observer: (e: RouterEvent) => { if (e.type === 'decision') events.set(e.requestId, e); } });
  let mode: RouterMode = 'enforce', refusal: string[] = [];
  let router;
  try { router = await load('enforce'); } catch (e) { if (!(e instanceof RouterLoadError && e.code === 'GATES_FAILED')) throw e; mode = 'shadow'; refusal = [...e.details]; router = await load('shadow'); }
  const items = c.checked.items;
  for (const [k, item] of items.entries()) await router.route({ text: item.text, requestId: `s${k}` });
  await router.flush();
  // Each head's own firing (the promise is about the head, before priority picks one route): the rules
  // and edge exactly as the router's edge tier evaluates them, the head scores from the same embeddings.
  const tier = createEdgeTier(read(c.dir, manifest.files.rules.key), edge, policy.priority, edge?.static ? await staticEncoder() : undefined);
  const vectors: Array<readonly number[]> = [];
  for (let k = 0; k < items.length; k += 256) vectors.push(...await encoder.embed(items.slice(k, k + 256).map((i) => i.text), { signal: new AbortController().signal }));
  const perHeadReport = evaluatePerHead(
    { heads: Object.fromEntries(Object.entries(classifier.heads as Record<string, { threshold: number; review_floor: number }>).map(([h, spec]) => [h, { threshold: spec.threshold, reviewFloor: spec.review_floor }])) },
    items.map((item, k) => ({ rules: tier.evaluate(item.text).findings, scores: scoreEmbedding(classifier, vectors[k]) as Record<string, number>, labels: item.truth })),
  );

  const routes: string[] = policy.routes.map((r: { id: string }) => r.id);
  const counts: Record<Pathway, number> = { rule: 0, edge: 0, settled: 0, model: 0, review: 0, none: 0, unavailable: 0 };
  const correct: Record<Pathway, number> = { ...counts };
  const examples: Record<Pathway, Array<{ text: string | null; to: string | null; ok: boolean | null; truth: string[] }>> = { rule: [], edge: [], settled: [], model: [], review: [], none: [], unavailable: [] };
  const perRoute = Object.fromEntries(routes.map((r) => [r, { routed: 0, hits: 0, positives: 0 }]));
  const candidates: Array<{ p: Pathway; order: number; ex: (typeof examples)['rule'][number] }> = [];
  for (const [k, item] of items.entries()) {
    const e = events.get(`s${k}`)!;
    const p = pathwayOf(e);
    counts[p]++;
    const truth = routes.filter((r) => item.truth[r] === 1);
    for (const r of truth) perRoute[r].positives++;
    let ok: boolean | null = null;
    if (e.outcome === 'route' && e.routeId) { perRoute[e.routeId].routed++; ok = item.truth[e.routeId] === 1; if (ok) perRoute[e.routeId].hits++; }
    else if (e.outcome === 'abstain') ok = truth.length === 0;
    if (ok) correct[p]++;
    // Examples: short, correct decisions; ToxicChat's toxic and jailbreak prompts are withheld.
    const withheld = c.key === 'toxicchat' && truth.length > 0;
    const text = item.text.replace(/\s+/g, ' ').trim();
    if ((ok ?? true) && text.length >= 20 && text.length <= 150) candidates.push({ p, order: stableHash(item.id), ex: { text: withheld ? null : text, to: e.routeId, ok, truth } });
  }
  candidates.sort((a, b) => a.order - b.order);
  for (const cand of candidates) if (examples[cand.p].length < 3) examples[cand.p].push(cand.ex);

  const heads = routes.map((r) => {
    const spec = classifier.heads[r];
    const criteria = evidence?.criteria?.heads?.[r] ?? {};
    const precisionPromise = criteria.minPrecision !== undefined || spec?.guarantee?.metric === 'precision';
    const target: number = criteria.minPrecision ?? criteria.minRecall ?? (spec?.guarantee ? 1 - spec.guarantee.alpha : NaN);
    const atRelease = evidence?.perHead?.heads?.[r];
    const pr = perRoute[r], hr = perHeadReport.heads[r]!;
    const checkedRecall = wilson(hr.recall.count, hr.recall.of), checkedPrecision = wilson(hr.precision.count, hr.precision.of), checkedFa = wilson(hr.falseAlarmRate.count, hr.falseAlarmRate.of);
    const met = precisionPromise ? (checkedPrecision ? checkedPrecision.rate >= target : null) : (checkedRecall ? checkedRecall.rate >= target : null);
    return {
      id: r, definition: info.routes[r] ?? '', promise: `${precisionPromise ? 'precision' : 'recall'} ≥ ${target}`, metric: precisionPromise ? 'precision' : 'recall',
      type: summary?.heads?.[r]?.type ?? spec?.type ?? 'linear', guarantee: spec?.guarantee?.kind ?? 'none',
      edge: edge?.heads?.[r] ? edge.heads[r].scorers as string[] : null,
      release: atRelease ? { recall: atRelease.recall?.rate ?? null, precision: atRelease.precision?.rate ?? null, falseAlarms: atRelease.falseAlarmRate?.rate ?? null, positives: atRelease.positives } : null,
      checked: { recall: checkedRecall, precision: checkedPrecision, falseAlarms: checkedFa, met },
      finalRoute: { recall: wilson(pr.hits, pr.positives), precision: wilson(pr.hits, pr.routed) },
    };
  });
  const emb = manifest.embedding ?? classifier.embedding;
  releases.push({
    key: c.key, title: info.title, blurb: info.blurb, name: c.name, releaseId: manifest.releaseId, manifest: deploy.manifestSha256.slice(0, 12),
    mode, refusal: summary?.enforcement ?? refusal,
    encoder: { model: String(emb.modelId ?? emb.model_id).split('/').pop(), layers: emb.layers ?? null, dimensions: emb.dimensions },
    edge: edge ? { scorers: [...new Set(Object.values(edge.heads).flatMap((h) => (h as { scorers: string[] }).scorers))] } : null,
    labelsSpent: summary?.labels_spent ?? null, heads, checkedOn: { description: c.checked.description, messages: items.length },
    pathways: counts, correct, examples,
    evidenceItems: evidence?.final?.items ?? null,
  });
  console.error(`${c.name}: ${items.length} messages routed (${mode}); ${Object.entries(counts).filter(([, v]) => v).map(([k, v]) => `${k} ${(100 * v / items.length).toFixed(1)}%`).join(', ')}`);
}

const data = { generated: new Date().toISOString().slice(0, 10), releases, refused: REFUSED };
mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, 'index.html'), page(JSON.stringify(data).replace(/</g, '\\u003c')));
console.error(`wrote ${join(OUT, 'index.html')}`);

function page(json: string): string {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Liquidau Routing Examples</title>
<meta name="description" content="Each example dataset's release, its certified classifier, and the routing pathways messages take.">
<style>
:root{--bg:#f7f7f4;--surface:#ffffff;--ink:#1c1f23;--muted:#5d6670;--line:#e2e3df;--accent:#2f6fd6;--rule:#7c4dce;--edge:#2f6fd6;--settled:#9aa3ad;--model:#1f9d6b;--review:#d88a1c;--none:#c8cdd2;--bad:#c2412d;--good:#1f9d6b;--chip:#eef1f5}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121417;--surface:#1b1e22;--ink:#e8eaed;--muted:#a0a8b1;--line:#2c3036;--accent:#6ea0ff;--rule:#a98af0;--edge:#6ea0ff;--settled:#6b7480;--model:#43c48f;--review:#f0a94a;--none:#4a5059;--bad:#f07a63;--good:#43c48f;--chip:#262a30}}
:root[data-theme="dark"]{--bg:#121417;--surface:#1b1e22;--ink:#e8eaed;--muted:#a0a8b1;--line:#2c3036;--accent:#6ea0ff;--rule:#a98af0;--edge:#6ea0ff;--settled:#6b7480;--model:#43c48f;--review:#f0a94a;--none:#4a5059;--bad:#f07a63;--good:#43c48f;--chip:#262a30}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1240px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:20px;margin:40px 0 4px}
.lede{color:var(--muted);max-width:760px;margin:0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(150px,100%),1fr));gap:12px;margin:24px 0 8px}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.stat b{display:block;font-size:24px;letter-spacing:-.01em}
.stat span{color:var(--muted);font-size:13px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px;margin-top:12px}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.tabs button{font:inherit;font-size:14px;border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:999px;padding:5px 14px;cursor:pointer}
.tabs button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.diagram-wrap{overflow-x:auto}
svg.flow{display:block;width:100%;min-width:760px;height:auto}
svg.flow text{fill:var(--ink);font-family:inherit}
svg.flow .muted{fill:var(--muted)}
svg.flow .node{fill:var(--surface);stroke:var(--line);stroke-width:1.5}
.note{color:var(--muted);font-size:13px;margin:8px 0 0}
.examples{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(180px,100%),1fr));gap:10px;margin-top:14px}
.ex{border:1px solid var(--line);border-radius:10px;padding:10px 12px;border-top:4px solid var(--c)}
.ex h4{margin:0 0 6px;font-size:13px}
.ex q{display:block;font-size:13px;color:var(--ink);margin:6px 0 2px;quotes:"\\201C" "\\201D"}
.ex small{color:var(--muted);font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(330px,100%),1fr));gap:16px;margin-top:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.bar{display:flex;height:6px;background:var(--chip)}
.bar i{display:block;height:100%}
.card .body{padding:16px 18px 18px;display:flex;flex-direction:column;gap:10px;flex:1}
.card h3{margin:0;font-size:19px;display:flex;align-items:center;justify-content:space-between;gap:8px}
.pill{font-size:12px;font-weight:600;border-radius:999px;padding:2px 10px;white-space:nowrap}
.pill.enforce{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.pill.shadow{background:color-mix(in srgb,var(--review) 20%,transparent);color:var(--review)}
.pill.refused{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}
.blurb{color:var(--muted);font-size:14px;margin:0}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chips span{background:var(--chip);border-radius:6px;padding:2px 8px;font-size:12.5px}
.big{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.big div{border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.big b{display:block;font-size:20px}
.big span{color:var(--muted);font-size:12px}
.head{border-top:1px solid var(--line);padding-top:10px}
.head .top{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.head .top b{font-size:15px}
.head p{margin:2px 0 6px;color:var(--muted);font-size:13px}
.head table{width:100%;border-collapse:collapse;font-size:13px}
.head td{padding:2px 0}.head .dim{color:var(--muted)}.head .sub{font-size:12px;padding-top:6px;text-align:left!important}
.head td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
.foot{margin-top:auto;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:8px}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
footer{margin-top:40px;color:var(--muted);font-size:13px}
</style>
</head>
<body>
<main>
<header>
<h1>Certified routing examples</h1>
<p class="lede">Each dataset below has a release built by @liquidau's libraries: mined rules and a cheap edge model settle what they can, certified classifier heads handle the rest, and anything uncertain goes to review. Every number on this page is read from the release files or measured by routing its labelled check file through the router.</p>
<div class="stats" id="stats"></div>
</header>

<section>
<h2>Routing pathways</h2>
<p class="lede">Where each message goes, with the share of the selected dataset's check file on each pathway, and real messages that took it.</p>
<div class="panel">
<div class="tabs" id="tabs" role="group" aria-label="Dataset"></div>
<div class="diagram-wrap"><svg class="flow" id="flow" viewBox="0 0 1200 430" role="img" aria-label="Routing pathways"></svg></div>
<p class="note" id="flow-note"></p>
<div class="examples" id="examples"></div>
</div>
</section>

<section>
<h2>Datasets and their classifiers</h2>
<p class="lede">The promise is what the release certifies for each head on its labelled sample, and ✓ means the head kept it on the check file. "At release" is the release's own held-out test split; the check file is a larger labelled file never used to build the release. Head figures are each head's own firing; the last line is final routing, after the policy's priority picks one route per message.</p>
<div class="cards" id="cards"></div>
</section>
<footer id="footer"></footer>
</main>
<script>
const DATA = ${json};
const PATHS = {
  rule: { label: 'Rule fired', color: 'var(--rule)', terminal: 'Routed at the edge' },
  edge: { label: 'Edge model fired', color: 'var(--edge)', terminal: 'Routed at the edge' },
  settled: { label: 'Dismissed at the edge', color: 'var(--settled)', terminal: 'No route, never embedded' },
  model: { label: 'Routed by a certified head', color: 'var(--model)', terminal: 'Routed by the model tier' },
  review: { label: 'Review queue', color: 'var(--review)', terminal: 'Review: Clef or a person' },
  none: { label: 'Below every head\\'s floor', color: 'var(--none)', terminal: 'No route' },
};
const BOXES = [
  { keys: ['rule', 'edge'], label: 'Routed at the edge', color: 'var(--edge)' },
  { keys: ['settled'], label: 'No route, never embedded', color: 'var(--settled)' },
  { keys: ['model'], label: 'Routed by a certified head', color: 'var(--model)' },
  { keys: ['review'], label: 'Review queue', color: 'var(--review)' },
  { keys: ['none', 'unavailable'], label: 'No route', color: 'var(--none)' },
];
const pct = (x, d = 1) => x == null || Number.isNaN(x) ? '–' : (100 * x).toFixed(d) + '%';
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const total = (r) => Object.values(r.pathways).reduce((a, b) => a + b, 0);
const share = (r, p) => r.pathways[p] / total(r);

// Header stats.
(() => {
  const msgs = DATA.releases.reduce((a, r) => a + total(r), 0);
  const edgeSettled = DATA.releases.reduce((a, r) => a + r.pathways.rule + r.pathways.edge + r.pathways.settled, 0);
  const heads = DATA.releases.flatMap((r) => r.heads), met = heads.filter((h) => h.checked.met).length;
  const stats = [[DATA.releases.length, 'releases built'], [msgs.toLocaleString('en'), 'check-file messages routed'], [pct(edgeSettled / msgs), 'settled without an embedding'], [met + ' / ' + heads.length, 'promises met on the check files']];
  document.getElementById('stats').innerHTML = stats.map(([b, s]) => '<div class="stat"><b>' + b + '</b><span>' + s + '</span></div>').join('');
})();

// Pathway diagram.
let current = DATA.releases.findIndex((r) => r.edge) >= 0 ? DATA.releases.findIndex((r) => r.edge) : 0;
const tabs = document.getElementById('tabs');
tabs.innerHTML = DATA.releases.map((r, i) => '<button type="button" data-i="' + i + '">' + esc(r.title) + '</button>').join('');
tabs.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { current = +b.dataset.i; drawFlow(); } });

function node(x, y, w, h, title, sub, color) {
  return '<g><rect class="node" x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="12"' + (color ? ' style="stroke:' + color + ';stroke-width:2"' : '') + '/>' +
    '<text x="' + (x + 14) + '" y="' + (y + 25) + '" font-size="15" font-weight="600">' + esc(title) + '</text>' +
    (sub || []).map((s, k) => '<text class="muted" x="' + (x + 14) + '" y="' + (y + 45 + k * 17) + '" font-size="12.5">' + esc(s) + '</text>').join('') + '</g>';
}
function link(x1, y1, x2, y2, s, color, label) {
  const w = s > 0 ? 2 + 30 * s : 1, mx = (x1 + x2) / 2;
  return '<path d="M' + x1 + ',' + y1 + ' C' + mx + ',' + y1 + ' ' + mx + ',' + y2 + ' ' + x2 + ',' + y2 + '" fill="none" stroke="' + color + '" stroke-opacity="' + (s > 0 ? .55 : .25) + '" stroke-width="' + w + '"/>' +
    (label ? '<text x="' + (mx) + '" y="' + ((y1 + y2) / 2 - w / 2 - 6) + '" font-size="13" font-weight="600" text-anchor="middle">' + label + '</text>' : '');
}
function drawFlow() {
  const r = DATA.releases[current];
  tabs.querySelectorAll('button').forEach((b, i) => b.setAttribute('aria-pressed', String(i === current)));
  const s = (p) => share(r, p);
  const fired = s('rule') + s('edge'), toModel = s('model') + s('review') + s('none') + s('unavailable');
  const edgeSub = r.edge ? ['rules · ' + r.edge.scorers.filter((x) => x !== 'rules').join(' · '), 'certified fire / dismiss, ~50-90 µs'] : ['mined rules only (no edge model)', 'certified rules, microseconds'];
  const enc = r.encoder.model + (r.encoder.layers ? ' (layers ' + r.encoder.layers.join('·') + ')' : '');
  const T = { fire: 20, settled: 105, model: 195, review: 280, none: 360 };
  let svg = '';
  svg += link(170, 215, 240, 215, 1, 'var(--muted)', '');
  svg += link(470, 190, 880, T.fire + 30, fired, 'var(--edge)', pct(fired));
  svg += link(470, 215, 880, T.settled + 30, s('settled'), 'var(--settled)', pct(s('settled')));
  svg += link(470, 240, 560, 280, toModel, 'var(--model)', pct(toModel));
  svg += link(800, 260, 880, T.model + 30, s('model'), 'var(--model)', pct(s('model')));
  svg += link(800, 285, 880, T.review + 30, s('review'), 'var(--review)', pct(s('review')));
  svg += link(800, 310, 880, T.none + 30, s('none') + s('unavailable'), 'var(--none)', pct(s('none') + s('unavailable')));
  svg += node(20, 180, 150, 70, 'Message', ['text in, one route out']);
  svg += node(240, 160, 230, 110, r.edge ? 'Edge tier' : 'Rules tier', edgeSub, 'var(--edge)');
  svg += node(560, 225, 240, 110, 'Model tier', [enc, 'certified heads, thresholds + floors'], 'var(--model)');
  const term = (y, color, title, sub) => node(880, y, 300, 62, title, [sub], color);
  svg += term(T.fire, 'var(--edge)', 'Routed at the edge', 'rule ' + pct(s('rule')) + ' · edge model ' + pct(s('edge')));
  svg += term(T.settled, 'var(--settled)', 'No route, never embedded', 'every head dismissed with certainty');
  svg += term(T.model, 'var(--model)', 'Routed by a certified head', 'score above the head\\'s threshold');
  svg += term(T.review, 'var(--review)', 'Review queue', 'between floor and threshold: Clef or a person');
  svg += term(T.none, 'var(--none)', 'No route', 'below every head\\'s floor');
  document.getElementById('flow').innerHTML = svg;
  document.getElementById('flow-note').textContent = r.title + ': ' + total(r).toLocaleString('en') + ' messages from ' + r.checkedOn.description + ', routed in ' + r.mode + ' mode. Line width is the share of messages.' + (r.pathways.unavailable ? ' ' + r.pathways.unavailable + ' unavailable (counted with no route).' : '');
  const ex = document.getElementById('examples');
  ex.innerHTML = BOXES.map((b) => {
    const n = b.keys.reduce((a, p) => a + r.pathways[p], 0), ok = b.keys.reduce((a, p) => a + r.correct[p], 0);
    const items = b.keys.flatMap((p) => r.examples[p] || []).slice(0, 3);
    const outcome = (e) => e.to ? '→ ' + esc(e.to) + (e.ok ? ' ✓' : '') : b.keys[0] === 'review' ? '→ review' + (e.truth.length ? ' · labelled ' + e.truth.map(esc).join(', ') : ' · no label') : '→ no route' + (e.ok ? ' ✓' : '');
    return '<div class="ex" style="--c:' + b.color + '"><h4>' + b.label + '</h4><small>' + pct(n / total(r)) + ' of messages' + (n && b.keys[0] !== 'review' ? ' · ' + pct(ok / n, 0) + ' correct' : '') + '</small>' +
      (items.length ? items.map((e) => '<q>' + (e.text === null ? '<em>[prompt withheld: labelled toxic]</em>' : esc(e.text)) + '</q><small>' + outcome(e) + '</small>').join('') : '<small><br>none in this file</small>') + '</div>';
  }).join('');
}
drawFlow();

// Cards.
const fmtRate = (w) => w ? pct(w.rate) + ' <span style="color:var(--muted)">(' + pct(w.lo) + '–' + pct(w.hi) + ')</span>' : '–';
const cards = DATA.releases.map((r) => {
  const n = total(r);
  const bar = Object.keys(PATHS).map((p) => '<i style="width:' + (100 * r.pathways[p] / n) + '%;background:' + PATHS[p].color + '" title="' + PATHS[p].label + ' ' + pct(r.pathways[p] / n) + '"></i>').join('');
  const edgeSettled = (r.pathways.rule + r.pathways.edge + r.pathways.settled) / n;
  const routed = r.pathways.rule + r.pathways.edge + r.pathways.model;
  const routedOk = r.correct.rule + r.correct.edge + r.correct.model;
  const heads = r.heads.map((h) => '<div class="head"><div class="top"><b>' + esc(h.id) + '</b><span class="' + (h.checked.met ? 'ok' : h.checked.met === false ? 'no' : '') + '">' + esc(h.promise) + (h.checked.met ? ' ✓' : h.checked.met === false ? ' ✗' : '') + '</span></div>' +
    '<p>' + esc(h.definition) + '</p><div class="chips"><span>' + esc(h.type) + ' head</span><span>' + esc(h.guarantee) + '</span>' + (h.edge ? '<span>edge: ' + esc(h.edge.join(' + ')) + '</span>' : '') + '</div>' +
    '<table><tr><td>' + h.metric + ' at release (test split)</td><td>' + (h.release ? pct(h.release[h.metric]) : '–') + '</td></tr>' +
    '<tr><td colspan="2" class="dim sub">on the check file (95% interval)</td></tr>' +
    '<tr><td>recall</td><td>' + fmtRate(h.checked.recall) + '</td></tr>' +
    '<tr><td>precision</td><td>' + fmtRate(h.checked.precision) + '</td></tr>' +
    '<tr><td>false alarms</td><td>' + fmtRate(h.checked.falseAlarms) + '</td></tr>' +
    '<tr><td class="dim">final routing</td><td class="dim">R ' + pct(h.finalRoute.recall?.rate) + ' · P ' + pct(h.finalRoute.precision?.rate) + '</td></tr></table></div>').join('');
  return '<article class="card"><div class="bar">' + bar + '</div><div class="body">' +
    '<h3>' + esc(r.title) + '<span class="pill ' + r.mode + '">' + (r.mode === 'enforce' ? 'Enforcing' : 'Shadow only') + '</span></h3>' +
    '<p class="blurb">' + esc(r.blurb) + '</p>' +
    '<div class="chips"><span>' + r.heads.length + ' route' + (r.heads.length > 1 ? 's' : '') + '</span><span>' + esc(r.encoder.model) + (r.encoder.layers ? ' · layers ' + r.encoder.layers.join('·') : '') + '</span><span>' + (r.edge ? 'edge: ' + esc(r.edge.scorers.join(' + ')) : 'rules only') + '</span>' + (r.labelsSpent ? '<span>' + r.labelsSpent.toLocaleString('en') + ' labels</span>' : '') + '</div>' +
    '<div class="big"><div><b>' + pct(edgeSettled, 0) + '</b><span>settled at the edge</span></div><div><b>' + pct(routed ? routedOk / routed : NaN, 0) + '</b><span>routed correctly</span></div><div><b>' + pct(r.pathways.review / n) + '</b><span>sent to review</span></div></div>' +
    heads +
    (r.mode === 'shadow' && r.refusal.length ? '<p class="note">Shadow only: ' + r.refusal.map(esc).join('; ') + '</p>' : '') +
    '<div class="foot">Checked on ' + n.toLocaleString('en') + ' messages: ' + esc(r.checkedOn.description) + '.<br>Release ' + esc(r.releaseId) + ' · manifest ' + esc(r.manifest) + '</div></div></article>';
});
const refused = DATA.refused.map((r) => '<article class="card"><div class="bar"><i style="width:100%;background:var(--bad);opacity:.5"></i></div><div class="body">' +
  '<h3>' + esc(r.title) + '<span class="pill refused">No release</span></h3><p class="blurb">' + esc(r.blurb) + '</p>' +
  r.routes.map((h) => '<div class="head"><div class="top"><b>' + esc(h.id) + '</b><span class="no">' + esc(h.promise) + '</span></div><p>' + esc(h.definition) + '</p><table><tr><td>lab result</td><td>' + esc(h.lab) + '</td></tr></table></div>').join('') +
  '<div class="foot">' + esc(r.reason) + ' Figures from the lab (liquidau-lab RESEARCH.md, stage EL), not a release.</div></div></article>');
document.getElementById('cards').innerHTML = cards.concat(refused).join('');
document.getElementById('footer').innerHTML = '<div class="legend">' + Object.values(PATHS).map((p) => '<span><i style="background:' + p.color + '"></i>' + p.label + '</span>').join('') + '</div><p>Generated ' + DATA.generated + ' by <code>npm run site</code> from the releases in <code>out/</code>.</p>';
</script>
</body>
</html>
`;
}
