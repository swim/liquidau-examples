/**
 * Local embedding models run through transformers.js, for comparing embedding models (e.g. the
 * benchmark). Pooling and prefixes follow each model card's guidance for classification or symmetric
 * tasks. Entries that fail to load are skipped and reported by the caller, not removed here.
 */
import { join } from 'node:path';

import { CachedEmbedder } from '@liquidau/embedding-classifier/embedder';
import { pipeline } from '@huggingface/transformers';

export interface ModelEntry {
  id: string;
  size: 'small' | 'base' | 'large';
  pooling: 'mean' | 'cls';
  /** Prepended to every text before embedding (part of what the vectors mean: keep it at runtime). */
  prefix?: string;
}

export const MODELS: ModelEntry[] = [
  // References, already cached for both datasets.
  { id: 'Xenova/all-MiniLM-L6-v2', size: 'small', pooling: 'mean' },
  { id: 'Xenova/all-mpnet-base-v2', size: 'base', pooling: 'mean' },
  // Small (about 33M parameters).
  { id: 'Xenova/bge-small-en-v1.5', size: 'small', pooling: 'cls' },
  { id: 'Xenova/gte-small', size: 'small', pooling: 'mean' },
  { id: 'Xenova/e5-small-v2', size: 'small', pooling: 'mean', prefix: 'query: ' },
  { id: 'Snowflake/snowflake-arctic-embed-s', size: 'small', pooling: 'cls' },
  // Base (about 110M).
  { id: 'Xenova/bge-base-en-v1.5', size: 'base', pooling: 'cls' },
  { id: 'Xenova/gte-base', size: 'base', pooling: 'mean' },
  { id: 'Xenova/e5-base-v2', size: 'base', pooling: 'mean', prefix: 'query: ' },
  { id: 'nomic-ai/nomic-embed-text-v1.5', size: 'base', pooling: 'mean', prefix: 'classification: ' },
  { id: 'Snowflake/snowflake-arctic-embed-m-v1.5', size: 'base', pooling: 'cls' },
  // Large (about 300M+).
  { id: 'Xenova/bge-large-en-v1.5', size: 'large', pooling: 'cls' },
  { id: 'mixedbread-ai/mxbai-embed-large-v1', size: 'large', pooling: 'cls' },
  { id: 'Xenova/e5-large-v2', size: 'large', pooling: 'mean', prefix: 'query: ' },
  { id: 'onnx-community/embeddinggemma-300m-ONNX', size: 'large', pooling: 'mean', prefix: 'task: classification | query: ' },
];

/**
 * A cached embedder for a registry entry. Tries 8-bit weights first, then full precision. The cache
 * file is per model (and per dataset via `cache`), matching makeEmbedder's names for the two
 * reference models, so their existing caches are reused.
 */
export async function makeModelEmbedder(entry: ModelEntry, options: { cache?: string; maxChars?: number; log?: (line: string) => void } = {}) {
  let extract;
  let dtype: 'q8' | 'fp32' = 'q8';
  try { extract = await pipeline('feature-extraction', entry.id, { dtype: 'q8' }); }
  catch { dtype = 'fp32'; extract = await pipeline('feature-extraction', entry.id, { dtype: 'fp32' }); }
  const embed = async (texts: string[]) => (await extract(texts.map((t) => (entry.prefix ?? '') + t), { pooling: entry.pooling, normalize: true })).tolist() as number[][];
  const dimensions = (await embed(['probe']))[0].length;
  const embedder = new CachedEmbedder({
    cachePath: join(import.meta.dirname, '..', '.cache', `embeddings-${options.cache ? `${options.cache}-` : ''}${entry.id.split('/').pop()}.json`),
    batchSize: 32, maxChars: options.maxChars, dimensions, log: options.log ?? (() => {}), embedBatch: embed,
  });
  return { embedder, embed, dimensions, dtype };
}
