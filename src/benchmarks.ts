/**
 * Three more benchmarks, as a TRAIN file (the traffic a round samples and labels) and a later or
 * held-out FUTURE file (where guarantees are checked), each message with its true labels:
 *
 *   ADE   ADE Corpus V2 classification (Hugging Face ade-benchmark-corpus/ade_corpus_v2, licence
 *         unknown, from public PubMed case reports): 23,516 sentences, head `ade` = describes an
 *         adverse drug event (29%). No official split: identical sentences grouped, groups split
 *         50/50 by a seeded hash.
 *   CFPB  US Consumer Financial Protection Bureau complaint narratives (CC0 snapshot of the public
 *         database, Hugging Face BEE-spoke-data/consumer-finance-complaints; the live database no
 *         longer publishes narratives). A REAL time split: 20,000 narratives from 2022 as the train
 *         file, 10,000 from July-December 2023 as the future. Heads: `fraud` = the issue or sub-issue
 *         names fraud, a scam, identity theft or unauthorised activity (keyword-defined, so it
 *         survives the 2023 renaming of products); `debt` = product "Debt collection". The database
 *         is not a statistical sample of consumers' experiences (CFPB's own caveat).
 *   ABCD  ASAPP's Action-Based Conversations Dataset (MIT): 10,042 customer-service chats. The text
 *         is the customer's opening turns, before the agent's reply. Heads: `access` = account access
 *         (recover password or username, reset 2FA); `refund` = initiate, check or update a refund.
 *         Official split: train file = train; future = dev + test.
 *
 * Data under data/ (not tracked): see each loader for where it comes from.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';

export interface Doc { id: string; text: string; group: string; truth: Record<string, 0 | 1>; sub?: string }
export interface Benchmark { train: Doc[]; future: Doc[] }

const DATA = join(import.meta.dirname, '..', 'data');
const norm = (t: string) => t.toLowerCase().normalize('NFKC').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
const hash = (s: string) => createHash('sha256').update(s).digest().readUInt32BE(0);
const need = (path: string, how: string) => { if (!existsSync(path)) throw new Error(`${path} is missing: ${how}`); return path; };

export function loadAde(): Benchmark {
  const rows = JSON.parse(readFileSync(need(join(DATA, 'ade', 'classification.json'), 'download Ade_corpus_v2_classification (see benchmarks.ts) to data/ade/classification.parquet and convert'), 'utf8')) as Array<{ text: string; label: 0 | 1 }>;
  const seen = new Set<string>(), docs: Doc[] = [];
  rows.forEach((r, k) => {
    const g = norm(r.text);
    if (!g || seen.has(g)) return; // identical sentences (the corpus lists one per drug-effect pair) count once
    seen.add(g);
    docs.push({ id: `ade-${k}`, text: r.text, group: g, truth: { ade: r.label ? 1 : 0 } });
  });
  return { train: docs.filter((d) => hash(d.group) % 2 === 0), future: docs.filter((d) => hash(d.group) % 2 === 1) };
}

export function loadCfpb(): Benchmark {
  const s = JSON.parse(readFileSync(need(join(DATA, 'cfpb', 'sample.json'), 'see benchmarks.ts: sampled from the CC0 snapshot'), 'utf8')) as Record<'train' | 'future', Array<{ id: string; date: string; product: string; issue: string; sub_issue: string | null; text: string }>>;
  const fraud = (r: { issue: string; sub_issue: string | null }) => (/fraud|scam|identity theft|unauthori[sz]ed/i.test(`${r.issue} / ${r.sub_issue ?? ''}`) ? 1 : 0);
  // Group by the first 200 normalised characters: templated narratives (common in credit reporting) count once.
  const doc = (r: (typeof s.train)[number]): Doc => ({ id: `cfpb-${r.id}`, text: r.text, group: norm(r.text).slice(0, 200), truth: { fraud: fraud(r), debt: r.product === 'Debt collection' ? 1 : 0 }, sub: r.product });
  const train = s.train.map(doc), trainGroups = new Set(train.map((d) => d.group));
  return { train, future: s.future.map(doc).filter((d) => !trainGroups.has(d.group)) };
}

export function loadAbcd(): Benchmark {
  const d = JSON.parse(gunzipSync(readFileSync(need(join(DATA, 'abcd', 'abcd_v1.1.json.gz'), 'download from github.com/asappresearch/abcd (data/abcd_v1.1.json.gz)'))).toString('utf8')) as Record<string, Array<{ convo_id: number; scenario: { flow: string; subflow: string }; original: Array<[string, string]> }>>;
  const opening = (turns: Array<[string, string]>) => {
    const out: string[] = [];
    for (const [who, text] of turns) {
      if (who === 'customer') out.push(text);
      else if (out.length) break; // the agent's reply after the customer's opening ends it
    }
    return out.join(' ');
  };
  const doc = (c: (typeof d.train)[number]): Doc => ({
    id: `abcd-${c.convo_id}`, text: opening(c.original), group: `abcd-${c.convo_id}`, sub: `${c.scenario.flow}/${c.scenario.subflow}`,
    truth: { access: c.scenario.flow === 'account_access' ? 1 : 0, refund: c.scenario.flow === 'product_defect' && c.scenario.subflow.startsWith('refund') ? 1 : 0 },
  });
  return { train: d.train.map(doc).filter((x) => x.text), future: [...d.dev, ...d.test].map(doc).filter((x) => x.text) };
}
