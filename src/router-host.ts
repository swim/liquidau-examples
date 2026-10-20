/**
 * Host pieces for @liquidau/router, which itself has no I/O: a filesystem release source, the
 * build-time package versions a release manifest records, and encoder adapters that implement an
 * exact EncoderIdentity for the example's embedders. Node only (node:fs, node:crypto).
 */
import { createHash } from 'node:crypto';
import { createReadStream, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { env } from '@huggingface/transformers';
import type { BuiltRelease, Encoder, EncoderIdentity } from '@liquidau/router';

import { GEMMA_MAX_TOKENS, GEMMA_MODEL, type GemmaEmbedder } from './embed-gemma.ts';
import type { LayerEmbedder } from './embed-layers.ts';

const ROOT = join(import.meta.dirname, '..');

export { fileSource } from './release-source.ts';

/**
 * Writes a built release to a directory, plus deploy.json: the manifest digest a deployment pins. In
 * production that digest comes from deployment configuration, not from next to the release.
 */
export function writeRelease(dir: string, built: BuiltRelease): void {
  mkdirSync(dir, { recursive: true });
  for (const [key, bytes] of built.files) writeFileSync(join(dir, key), bytes);
  writeFileSync(join(dir, 'deploy.json'), JSON.stringify({ manifestKey: built.manifestKey, manifestSha256: built.manifestSha256 }, null, 1));
}

/** Exact installed versions of the Liquidau packages a release manifest records. */
export function packageVersions(): Record<string, string> {
  const names = ['@liquidau/embedding-classifier', '@liquidau/rule-miner', '@liquidau/router', '@liquidau/text-preprocessing'];
  return Object.fromEntries(names.map((n) => [n, (JSON.parse(readFileSync(join(ROOT, 'node_modules', n, 'package.json'), 'utf8')) as { version: string }).version]));
}

/** SHA-256 of a file's bytes, streamed (model files are hundreds of megabytes). */
export async function sha256File(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest('hex');
}

const tokenizerDigest = async (model: string) => `${model}/tokenizer.json@sha256:${await sha256File(join(env.cacheDir!, model, 'tokenizer.json'))}`;

/** Wraps a batch function as a router Encoder: an already-aborted signal rejects, as the contract requires. */
function encoder(identity: EncoderIdentity, embedBatch: (texts: readonly string[]) => Promise<number[][]>): Encoder {
  const frozen = Object.freeze({ ...identity, layers: identity.layers ? Object.freeze([...identity.layers]) as number[] : null });
  return {
    identity: frozen,
    async embed(texts, { signal }) {
      if (signal.aborted) throw new Error('aborted');
      const rows = await embedBatch(texts);
      if (signal.aborted) throw new Error('aborted');
      return rows;
    },
  };
}

/**
 * The multi-layer embedder (src/embed-layers.ts): layers 4, 8 and 12 of all-mpnet-base-v2, each
 * mean-pooled and unit-normalised, then concatenated. Its revision is the ONNX export's content digest,
 * so a release is served only by the exact model bytes it was trained on. Truncation is declared:
 * the text is cut to maxChars (surrogate-safe), then to 383 tokens plus the closing </s>.
 */
export async function layerEncoder(embedder: LayerEmbedder, modelPath: string, embedBatch: (texts: readonly string[]) => Promise<number[][]> = (t) => embedder.embed(t)): Promise<Encoder> {
  const s = embedder.spec;
  return encoder({
    schema: 'liquidau-encoder/1', modelId: s.model_id, revision: `sha256:${await sha256File(modelPath)}`, dimensions: s.dimensions, precision: s.precision!,
    inputType: null, layers: [...s.layers!], pooling: s.pooling!, normalization: 'per-layer-unit', tokenizerRevision: await tokenizerDigest('Xenova/all-mpnet-base-v2'),
    maxChars: s.max_chars ?? null, maxTokens: 384, truncation: 'utf16-chars-then-tokens-keep-eos/1',
  }, embedBatch);
}

/**
 * EmbeddingGemma (src/embed-gemma.ts): the prompt is the input type, the vectors are the model's own
 * sentence embedding, unit length. The revision digests the ONNX graph and its external weights file
 * together, in that order, so a release is served only by the exact model bytes it was trained on.
 */
export async function gemmaEncoder(embedder: GemmaEmbedder, embedBatch: (texts: readonly string[]) => Promise<number[][]> = (t) => embedder.embed(t)): Promise<Encoder> {
  const s = embedder.spec, dir = join(env.cacheDir!, GEMMA_MODEL, 'onnx');
  const hash = createHash('sha256');
  for (const f of ['model_quantized.onnx', 'model_quantized.onnx_data']) for await (const chunk of createReadStream(join(dir, f))) hash.update(chunk as Buffer);
  return encoder({
    schema: 'liquidau-encoder/1', modelId: s.model_id, revision: `sha256:${hash.digest('hex')}`, dimensions: s.dimensions, precision: s.precision!,
    inputType: s.input_type!, layers: null, pooling: 'model-sentence-embedding', normalization: 'unit', tokenizerRevision: await tokenizerDigest(GEMMA_MODEL),
    maxChars: s.max_chars ?? null, maxTokens: GEMMA_MAX_TOKENS, truncation: 'utf16-chars-then-transformers-js-max-length/1',
  }, embedBatch);
}

/**
 * The walkthrough's transformers.js embedder (src/embed.ts): mean pooling, unit length, 8-bit weights.
 * transformers.js truncates silently at the model's 512 tokens, which the identity declares.
 */
export async function transformersEncoder(model: string, dimensions: number, embedBatch: (texts: readonly string[]) => Promise<number[][]>): Promise<Encoder> {
  return encoder({
    schema: 'liquidau-encoder/1', modelId: model, revision: `sha256:${await sha256File(join(env.cacheDir!, model, 'onnx', 'model_quantized.onnx'))}`, dimensions, precision: 'q8',
    inputType: null, layers: null, pooling: 'mean', normalization: 'unit', tokenizerRevision: await tokenizerDigest(model),
    maxChars: null, maxTokens: 512, truncation: 'transformers-js-max-length/1',
  }, embedBatch);
}
