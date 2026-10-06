/**
 * Reference multi-layer embedder: all-mpnet-base-v2 through the ONNX export written by
 * scripts/export_layers_onnx.py (`npm run export:onnx`), whose graph outputs every layer's mean-pooled, unit-normalised
 * state (`pooled`, [batch, 12, 768]). The features are the chosen layers concatenated, exactly as the
 * heads were trained on them; `spec` is the EmbeddingSpec to record in the artifact and to compare at
 * start-up (embedding-classifier's checkEmbeddingSpec).
 *
 * Node only (onnxruntime-node). Tokenising follows sentence-transformers: truncate the text to
 * `maxChars` characters (surrogate-safe), then to 384 tokens KEEPING the closing </s> - transformers.js's
 * own truncation drops it on over-long inputs, which changes the embedding of those texts.
 */
import { AutoTokenizer } from '@huggingface/transformers';
import { truncateText, type EmbeddingSpec } from '@liquidau/embedding-classifier';
import * as ort from 'onnxruntime-node';

const MAX_TOKENS = 384;
const PAD = 1n; // mpnet's <pad>

export interface LayerEmbedder {
  spec: EmbeddingSpec;
  embed(texts: readonly string[]): Promise<number[][]>;
}

export async function makeLayerEmbedder(modelPath: string, options: { layers?: number[]; maxChars?: number; precision?: string; threads?: number } = {}): Promise<LayerEmbedder> {
  const layers = options.layers ?? [4, 8, 12];
  const maxChars = options.maxChars ?? 2000;
  const tokenizer = await AutoTokenizer.from_pretrained('Xenova/all-mpnet-base-v2');
  const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'], ...(options.threads ? { intraOpNumThreads: options.threads } : {}) });
  const spec: EmbeddingSpec = { model_id: 'sentence-transformers/all-mpnet-base-v2', dimensions: 768 * layers.length, normalize: true, layers, pooling: 'mean', layer_normalize: true, precision: options.precision ?? 'fp32', max_chars: maxChars };
  return {
    spec,
    async embed(texts) {
      if (!texts.length) return [];
      // Encode each text in full, then cut to 383 tokens + </s>, then pad: Python's truncation.
      const encoded = texts.map((t) => {
        const ids = Array.from((tokenizer(truncateText(t, maxChars)) as { input_ids: { data: BigInt64Array } }).input_ids.data);
        return ids.length > MAX_TOKENS ? [...ids.slice(0, MAX_TOKENS - 1), ids[ids.length - 1]] : ids;
      });
      const len = Math.max(...encoded.map((ids) => ids.length));
      const ids = new BigInt64Array(texts.length * len).fill(PAD), mask = new BigInt64Array(texts.length * len);
      encoded.forEach((e, b) => e.forEach((v, j) => { ids[b * len + j] = v; mask[b * len + j] = 1n; }));
      const feeds = { input_ids: new ort.Tensor('int64', ids, [texts.length, len]), attention_mask: new ort.Tensor('int64', mask, [texts.length, len]) };
      const out = (await session.run(feeds)).pooled;
      const data = out.data as Float32Array, [batch, nLayers, width] = out.dims as number[];
      return Array.from({ length: batch }, (_, b) => layers.flatMap((l) => {
        if (l < 1 || l > nLayers) throw new Error(`layer ${l} is outside the export's 1..${nLayers}`);
        const off = (b * nLayers + (l - 1)) * width;
        return Array.from(data.subarray(off, off + width));
      }));
    },
  };
}
