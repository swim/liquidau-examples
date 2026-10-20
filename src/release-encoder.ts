/**
 * The release pipeline's encoders: 'gemma' (the default, src/embed-gemma.ts) or 'mpnet-layers'
 * (all-mpnet-base-v2 layers 4, 8 and 12 through the ONNX export, src/embed-layers.ts). Each comes with
 * its own embedding cache under .cache/release/, so training, the simulation and the checks embed each
 * text once. `releaseEmbedderFor` picks the one a release's classifier was trained with.
 */
import { join } from 'node:path';

import type { EmbeddingSpec } from '@liquidau/embedding-classifier';
import { CachedEmbedder } from '@liquidau/embedding-classifier/embedder';
import type { Encoder } from '@liquidau/router';

import { GEMMA_MODEL, makeGemmaEmbedder } from './embed-gemma.ts';
import { makeLayerEmbedder } from './embed-layers.ts';
import { gemmaEncoder, layerEncoder } from './router-host.ts';

const ROOT = join(import.meta.dirname, '..');

export type ReleaseEncoderName = 'gemma' | 'mpnet-layers';

export interface ReleaseEncoderOptions {
  /** Default 'gemma'. */
  encoder?: ReleaseEncoderName;
  /** 'mpnet-layers' only: the ONNX export (default .cache/onnx/mpnet-layers.onnx) and its layers (default [4, 8, 12]). */
  model?: string;
  layers?: number[];
  /** Characters kept per text before tokenising (default 2000). */
  maxChars?: number;
  log?: (line: string) => void;
}

export interface ReleaseEmbedder {
  spec: EmbeddingSpec;
  /** Cached: each distinct text is embedded once per encoder. */
  embedMany(texts: readonly string[]): Promise<number[][]>;
  /** The router encoder for this embedder, answering from the same cache. */
  encoder(): Promise<Encoder>;
}

export async function releaseEmbedder(options: ReleaseEncoderOptions = {}): Promise<ReleaseEmbedder> {
  const maxChars = options.maxChars ?? 2000;
  const cached = (name: string, spec: EmbeddingSpec, embed: (texts: string[]) => Promise<number[][]>, batchSize: number) => new CachedEmbedder({
    cachePath: join(ROOT, '.cache', 'release', name), embedBatch: embed, batchSize, maxChars: spec.max_chars, dimensions: spec.dimensions, format: 'binary', log: options.log ?? (() => {}),
  });
  if ((options.encoder ?? 'gemma') === 'gemma') {
    const inner = await makeGemmaEmbedder({ maxChars });
    const cache = cached('emb-gemma', inner.spec, (t) => inner.embed(t), 16);
    return { spec: inner.spec, embedMany: (t) => cache.embedMany([...t]), encoder: () => gemmaEncoder(inner, (t) => cache.embedMany([...t])) };
  }
  const layers = options.layers ?? [4, 8, 12];
  const model = options.model ?? join(ROOT, '.cache', 'onnx', 'mpnet-layers.onnx');
  const inner = await makeLayerEmbedder(model, { layers, maxChars });
  const cache = cached(`emb-${layers.join('-')}`, inner.spec, (t) => inner.embed(t), 16);
  return { spec: inner.spec, embedMany: (t) => cache.embedMany([...t]), encoder: () => layerEncoder(inner, model, (t) => cache.embedMany([...t])) };
}

/** The release embedder a classifier artifact's recorded spec was trained with. */
export function releaseEmbedderFor(spec: EmbeddingSpec, log?: (line: string) => void): Promise<ReleaseEmbedder> {
  return spec.model_id === GEMMA_MODEL
    ? releaseEmbedder({ encoder: 'gemma', maxChars: spec.max_chars, log })
    : releaseEmbedder({ encoder: 'mpnet-layers', layers: spec.layers, maxChars: spec.max_chars, log });
}
