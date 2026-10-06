/**
 * ToxicChat (LMSYS, CC BY-NC 4.0), version 0124: 10,165 real user prompts sent to the Vicuna demo,
 * labelled for toxicity (7.3%) and jailbreaking (2.0%). https://huggingface.co/datasets/lmsys/toxic-chat
 *
 * WARNING: the prompts include offensive and sexual text.
 *
 * English only by default. The embedding models used here are English models, so non-English
 * prompts (about 4.4%: Portuguese, Spanish, French and others, nearly all labelled non-toxic) would
 * measure the model's language coverage rather than the methods under test. Language is detected
 * with eld (https://github.com/nitotm/efficient-language-detector-js); spot checks put it right on
 * about 30 of 31 flagged prompts.
 *
 * Unlike Banking77 the labels are binary and the prompts arrive at their natural prevalence, so no
 * scenario has to be constructed: the official train file plays the traffic we sample and label,
 * and the official test file is a fully labelled "production" period to check the guarantees on.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { eld } from 'eld';

import { parseCsv } from './data.ts';

export const TC_HEADS = ['toxic', 'jailbreak'] as const;
export type TcHead = (typeof TC_HEADS)[number];

export interface Prompt {
  id: string;
  text: string;
  /** Exact duplicates (after normalising case and whitespace) share a group. */
  group: string;
  truth: Record<TcHead, 0 | 1>;
  /** False where the dataset's label came from its model-assisted pass rather than a person. */
  humanAnnotated: boolean;
  split: 'train' | 'test';
  /** ISO 639-1 code from eld ('' when it can't tell). */
  language: string;
  /**
   * For toxic prompts, the OpenAI Moderation category that scored highest (sexual, harassment,
   * violence, or other). A model's output, so usable as a weak training label only.
   */
  subtype?: string;
}

/** Texts are truncated to this before embedding (the model reads at most 384 tokens anyway). */
export const MAX_CHARS = 2000;

const SOURCE = 'https://huggingface.co/datasets/lmsys/toxic-chat/resolve/main/data/0124';
const DATA = join(import.meta.dirname, '..', 'data', 'toxicchat');

async function fetchCached(split: 'train' | 'test'): Promise<string> {
  mkdirSync(DATA, { recursive: true });
  const path = join(DATA, `${split}.csv`);
  if (!existsSync(path)) {
    const res = await fetch(`${SOURCE}/toxic-chat_annotation_${split}.csv`);
    if (!res.ok) throw new Error(`downloading ToxicChat ${split}: HTTP ${res.status}`);
    writeFileSync(path, await res.text());
  }
  return readFileSync(path, 'utf8');
}

const SUBTYPES = new Set(['sexual', 'harassment', 'violence']);
function subtypeOf(moderation: string): string {
  try {
    const scores = JSON.parse(moderation) as Array<[string, number]>;
    const top = scores.reduce((a, b) => (b[1] > a[1] ? b : a))[0].split('/')[0];
    return SUBTYPES.has(top) ? top : 'other';
  } catch {
    return 'other';
  }
}

let eldLoaded = false;
/** A prompt counts as non-English only when eld names another language and calls it reliable. */
async function languageOf(text: string): Promise<string> {
  // The default import ships without a database; 'medium' is loaded once on first use.
  if (!eldLoaded) { await (eld as unknown as { load(size: string): Promise<void> }).load('medium'); eldLoaded = true; }
  const d = eld.detect(text);
  return d.language && d.language !== 'en' && !d.isReliable() ? 'en?' : d.language;
}

export async function loadToxicChat(options: { englishOnly?: boolean } = {}): Promise<Prompt[]> {
  const { englishOnly = true } = options;
  const groupOf = new Map<string, string>();
  const out: Prompt[] = [];
  for (const split of ['train', 'test'] as const) {
    const [header, ...rows] = parseCsv(await fetchCached(split));
    const col = (name: string) => { const i = header.indexOf(name); if (i < 0) throw new Error(`ToxicChat ${split}: no column ${name}`); return i; };
    const [id, text, human, toxic, jailbreak, moderation] = ['conv_id', 'user_input', 'human_annotation', 'toxicity', 'jailbreaking', 'openai_moderation'].map(col);
    for (const r of rows) {
      const t = r[text].trim();
      if (!t) continue;
      const language = await languageOf(t);
      if (englishOnly && language !== 'en' && language !== 'en?' && language !== '') continue;
      const key = t.toLowerCase().replace(/\s+/g, ' ');
      if (!groupOf.has(key)) groupOf.set(key, `t${groupOf.size}`);
      out.push({
        id: `tc-${split}-${r[id]}`, text: t, group: groupOf.get(key)!, split, language, humanAnnotated: r[human] === 'True',
        truth: { toxic: r[toxic] === '1' ? 1 : 0, jailbreak: r[jailbreak] === '1' ? 1 : 0 },
        ...(r[toxic] === '1' ? { subtype: subtypeOf(r[moderation]) } : {}),
      });
    }
  }
  return out;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const everything = await loadToxicChat({ englishOnly: false });
  const dropped = everything.filter((p) => !['en', 'en?', ''].includes(p.language));
  console.log(`non-English (dropped by default): ${dropped.length} of ${everything.length}, toxic ${dropped.filter((p) => p.truth.toxic).length}, jailbreak ${dropped.filter((p) => p.truth.jailbreak).length}`);
  const all = await loadToxicChat();
  for (const split of ['train', 'test'] as const) {
    const s = all.filter((p) => p.split === split);
    const rate = (h: TcHead) => `${(100 * s.filter((p) => p.truth[h]).length / s.length).toFixed(2)}%`;
    console.log(`${split}: ${s.length} prompts, toxic ${rate('toxic')}, jailbreak ${rate('jailbreak')}, human-annotated ${s.filter((p) => p.humanAnnotated).length}, groups ${new Set(s.map((p) => p.group)).size}`);
  }
  const shared = new Set(all.filter((p) => p.split === 'train').map((p) => p.group));
  console.log(`groups in both splits: ${new Set(all.filter((p) => p.split === 'test' && shared.has(p.group)).map((p) => p.group)).size}`);
  console.log(`jailbreak but not toxic: ${all.filter((p) => p.truth.jailbreak && !p.truth.toxic).length}`);
}
