/**
 * Banking77 (PolyAI, CC BY 4.0): 13,083 online-banking customer queries labelled with 77 intents.
 * https://github.com/PolyAI-LDN/task-specific-datasets
 *
 * Two binary heads are defined over the intents:
 *   urgent  security and fraud - a miss is costly (recall head)
 *   refund  refund requests - a false alarm is costly (precision head)
 *
 * Banking77 is roughly balanced across intents, so "urgent" would be ~8% of it - far more than real
 * traffic. The scenario therefore builds a realistic stream: urgent and refund messages are
 * subsampled to `prevalence` of the traffic, and the rest become a historical, enriched labelled set
 * (escalations that were collected over time) - training data only, never an evaluation set.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { seededRandom } from '@liquidau/solvers';

export const HEADS = ['urgent', 'refund'] as const;
export type Head = (typeof HEADS)[number];

export const INTENTS: Record<Head, readonly string[]> = {
  urgent: ['compromised_card', 'lost_or_stolen_card', 'lost_or_stolen_phone', 'card_payment_not_recognised', 'cash_withdrawal_not_recognised', 'direct_debit_payment_not_recognised'],
  refund: ['request_refund', 'Refund_not_showing_up'],
};

export interface Message {
  id: string;
  text: string;
  intent: string;
  /** Exact duplicates (after normalising case and whitespace) share a group. */
  group: string;
  /** Ground truth per head - the simulated reviewer's answer. */
  truth: Record<Head, 0 | 1>;
  /** Banking77's own split, for benchmarks that use it (MTEB evaluates on the official test file). */
  split: 'train' | 'test';
}

const SOURCE = 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data';
const DATA = join(import.meta.dirname, '..', 'data');

/** Minimal RFC 4180 CSV parsing: quoted fields, doubled quotes, commas and newlines inside quotes. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.some((f) => f !== '')) rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  return rows;
}

async function fetchCached(name: string): Promise<string> {
  mkdirSync(DATA, { recursive: true });
  const path = join(DATA, name);
  if (!existsSync(path)) {
    const res = await fetch(`${SOURCE}/${name}`);
    if (!res.ok) throw new Error(`downloading ${name}: HTTP ${res.status}`);
    writeFileSync(path, await res.text());
  }
  return readFileSync(path, 'utf8');
}

export async function loadBanking77(): Promise<Message[]> {
  const train = parseCsv(await fetchCached('train.csv')).slice(1), test = parseCsv(await fetchCached('test.csv')).slice(1);
  const rows = [...train, ...test];
  const groupOf = new Map<string, string>();
  return rows.map(([text, intent], i) => {
    const key = text.trim().toLowerCase().replace(/\s+/g, ' ');
    if (!groupOf.has(key)) groupOf.set(key, `g${groupOf.size}`);
    return {
      id: `b77-${i}`, text: text.trim(), intent, group: groupOf.get(key)!, split: i < train.length ? 'train' : 'test',
      truth: { urgent: INTENTS.urgent.includes(intent) ? 1 : 0, refund: INTENTS.refund.includes(intent) ? 1 : 0 },
    };
  });
}

export interface Scenario {
  /** Historical labelled messages: the subsampled-away positives plus some negatives. Train only. */
  historical: Message[];
  /** Unlabelled traffic for mining rules (backgroundUse 'veto'). */
  veto: Message[];
  /** Unlabelled traffic for certifying rules (backgroundUse 'certify'). */
  certify: Message[];
  /** Unlabelled traffic for threshold budgets and as the drift reference (backgroundUse 'budget'). */
  budget: Message[];
  /** Held-out traffic standing in for a later live window. */
  live: Message[];
  /** The sampling frame for labelled train / calibration / test sets. */
  frame: Message[];
}

/**
 * A traffic stream at realistic prevalence, split into disjoint parts (by group, so no paraphrase
 * group straddles two roles).
 */
export function buildScenario(messages: readonly Message[], options: { seed: number; prevalence: Record<Head, number>; historicalNegatives: number; sizes: { veto: number; certify: number; budget: number; live: number } }): Scenario {
  const { seed, prevalence, historicalNegatives, sizes } = options;
  const rand = seededRandom(seed);
  const shuffle = <T>(xs: T[]) => { for (let i = xs.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [xs[i], xs[j]] = [xs[j], xs[i]]; } return xs; };
  // Whole groups move together.
  const groups = new Map<string, Message[]>();
  for (const m of messages) (groups.get(m.group) ?? groups.set(m.group, []).get(m.group)!).push(m);
  const kind = (g: Message[]): Head | 'other' => (g[0].truth.urgent ? 'urgent' : g[0].truth.refund ? 'refund' : 'other');
  const byKind: Record<Head | 'other', Message[][]> = { urgent: [], refund: [], other: [] };
  for (const g of shuffle([...groups.values()])) byKind[kind(g)].push(g);

  const historical: Message[][] = byKind.other.splice(0, historicalNegatives);
  const negatives = byKind.other;
  const traffic = negatives.length / (1 - prevalence.urgent - prevalence.refund);
  const keep = { urgent: Math.round(traffic * prevalence.urgent), refund: Math.round(traffic * prevalence.refund) };
  const stream = [...negatives];
  for (const h of HEADS) {
    stream.push(...byKind[h].slice(0, keep[h]));
    historical.push(...byKind[h].slice(keep[h]));
  }
  shuffle(stream);
  const take = (n: number) => stream.splice(0, n).flat();
  const veto = take(sizes.veto), certify = take(sizes.certify), budget = take(sizes.budget), live = take(sizes.live);
  return { historical: historical.flat(), veto, certify, budget, live, frame: stream.flat() };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const all = await loadBanking77();
  console.log(`${all.length} messages, ${new Set(all.map((m) => m.intent)).size} intents, ${new Set(all.map((m) => m.group)).size} groups`);
}
