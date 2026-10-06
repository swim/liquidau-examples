/**
 * Sentence embeddings computed locally via transformers.js (no API key), cached on disk by
 * embedding-classifier's CachedEmbedder so each distinct text is embedded once per model. The spec
 * is recorded in the artifact: runtime must embed exactly the same way.
 */
import { join } from 'node:path';

import type { EmbeddingSpec } from '@liquidau/embedding-classifier';
import { CachedEmbedder } from '@liquidau/embedding-classifier/embedder';
import { pipeline } from '@huggingface/transformers';

/**
 * The model the example ships with. In the stage 8 comparison it cut urgent's false alarms to about
 * a third of MiniLM's under the same recall guarantee, for ~4x slower embedding and 768 dimensions.
 */
export const MODEL = 'Xenova/all-mpnet-base-v2';
/** A smaller, faster model for comparison. */
export const SMALL_MODEL = 'Xenova/all-MiniLM-L6-v2';
const DIMENSIONS: Record<string, number> = { [MODEL]: 768, [SMALL_MODEL]: 384 };

export const embeddingSpec = (model: string): EmbeddingSpec => ({ model_id: model, dimensions: DIMENSIONS[model], normalize: true });
export const EMBEDDING = embeddingSpec(MODEL);

/**
 * `cache` names the cache file, so datasets embedded while another process uses the default cache
 * don't write over each other (two writers of one file can drop each other's newest entries).
 * `maxChars` truncates long texts; keep it identical at runtime.
 */
export async function makeEmbedder(log: (line: string) => void = () => {}, model = MODEL, options: { cache?: string; maxChars?: number } = {}) {
  if (!DIMENSIONS[model]) throw new Error(`unknown model ${model}`);
  const extract = await pipeline('feature-extraction', model, { dtype: 'q8' });
  return new CachedEmbedder({
    cachePath: join(import.meta.dirname, '..', '.cache', `embeddings-${options.cache ? `${options.cache}-` : ''}${model.split('/').pop()}.json`),
    batchSize: 64,
    maxChars: options.maxChars,
    dimensions: DIMENSIONS[model],
    log,
    embedBatch: async (texts) => {
      const out = await extract(texts, { pooling: 'mean', normalize: true });
      return out.tolist() as number[][];
    },
  });
}
