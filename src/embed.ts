/**
 * Sentence embeddings computed locally via transformers.js (no API key), cached on disk by
 * embedding-classifier's CachedEmbedder so each distinct text is embedded once per model. The spec
 * is recorded in the artifact: runtime must embed exactly the same way.
 */
import { join } from 'node:path';

import type { EmbeddingSpec } from '@liquidau/embedding-classifier';
import { CachedEmbedder } from '@liquidau/embedding-classifier/embedder';
import type { Encoder } from '@liquidau/router';
import { pipeline } from '@huggingface/transformers';

import { GEMMA_DIMENSIONS, GEMMA_MODEL, gemmaSpec, makeGemmaEmbedder } from './embed-gemma.ts';
import { gemmaEncoder, transformersEncoder } from './router-host.ts';

/**
 * The model the examples ship with: EmbeddingGemma (src/embed-gemma.ts). In liquidau-lab stage EG it cut
 * Banking77 `urgent`'s false alarms from 1.02% to 0.25% and ToxicChat `toxic`'s from 5.91% to 4.70% under
 * the same guarantees, but lost one `toxic` guarantee and missed the stage's pre-registered rule; it
 * costs about ten times mpnet per message. Gemma Terms of Use apply.
 */
export const MODEL = GEMMA_MODEL;
/** The previous default, about ten times faster: the walkthrough's comparison model. */
export const MPNET_MODEL = 'Xenova/all-mpnet-base-v2';
/** A small, fast model (the benchmark's reference). */
export const MINILM_MODEL = 'Xenova/all-MiniLM-L6-v2';
const DIMENSIONS: Record<string, number> = { [MODEL]: GEMMA_DIMENSIONS, [MPNET_MODEL]: 768, [MINILM_MODEL]: 384 };
const DEFAULT_MAX_CHARS = 2000;

/**
 * The spec the artifact records. EmbeddingGemma: its prompt as the input type and its character limit.
 * The others: transformers.js mean pooling, unit length, 8-bit ('q8') weights.
 */
export const embeddingSpec = (model: string): EmbeddingSpec =>
  model === GEMMA_MODEL ? gemmaSpec(DEFAULT_MAX_CHARS) : { model_id: model, dimensions: DIMENSIONS[model], normalize: true, precision: 'q8' };
export const EMBEDDING = embeddingSpec(MODEL);

/**
 * `cache` names the cache file, so datasets embedded while another process uses the default cache
 * don't write over each other (two writers of one file can drop each other's newest entries).
 * `maxChars` truncates long texts; keep it identical at runtime (EmbeddingGemma records it: 2000).
 */
export async function makeEmbedder(log: (line: string) => void = () => {}, model = MODEL, options: { cache?: string; maxChars?: number } = {}) {
  if (!DIMENSIONS[model]) throw new Error(`unknown model ${model}`);
  let embedBatch: (texts: string[]) => Promise<number[][]>;
  if (model === GEMMA_MODEL) {
    const maxChars = options.maxChars ?? DEFAULT_MAX_CHARS;
    if (maxChars !== DEFAULT_MAX_CHARS) throw new Error(`${model}: the recorded spec keeps ${DEFAULT_MAX_CHARS} characters, not ${maxChars}`);
    const gemma = await makeGemmaEmbedder({ maxChars });
    embedBatch = (texts) => gemma.embed(texts);
  } else {
    const extract = await pipeline('feature-extraction', model, { dtype: 'q8' });
    embedBatch = async (texts) => (await extract(texts, { pooling: 'mean', normalize: true })).tolist() as number[][];
  }
  return new CachedEmbedder({
    cachePath: join(import.meta.dirname, '..', '.cache', `embeddings-${options.cache ? `${options.cache}-` : ''}${model.split('/').pop()}.json`),
    batchSize: model === GEMMA_MODEL ? 16 : 64,
    maxChars: model === GEMMA_MODEL ? options.maxChars ?? DEFAULT_MAX_CHARS : options.maxChars,
    dimensions: DIMENSIONS[model],
    log,
    embedBatch,
  });
}

/** The router encoder for a walkthrough embedder (makeEmbedder's), answering through `embedBatch`. */
export function walkthroughEncoder(embedBatch: (texts: readonly string[]) => Promise<number[][]>, model = MODEL): Promise<Encoder> {
  return model === GEMMA_MODEL
    ? gemmaEncoder({ spec: embeddingSpec(model), embed: embedBatch }, embedBatch)
    : transformersEncoder(model, DIMENSIONS[model], embedBatch);
}
