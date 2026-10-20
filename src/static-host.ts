/**
 * The edge tier's static embedding model (Model2Vec minishlab/potion-base-8M: tokenizer.json and
 * model.safetensors in .cache/static/potion-base-8M, from Hugging Face), through text-preprocessing's
 * pure-TypeScript WordPiece and createStaticEmbedder, so the rules (edge) function needs no inference
 * engine. Loaded once per process.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { createStaticEmbedder, type StaticEmbedder } from '@liquidau/text-preprocessing';

export const STATIC_MODEL = { id: 'minishlab/potion-base-8M', dir: join(import.meta.dirname, '..', '.cache', 'static', 'potion-base-8M') };

let loaded: Promise<StaticEmbedder> | null = null;

/** The static encoder (one per process). */
export function staticEncoder(): Promise<StaticEmbedder> {
  loaded ??= (async () => {
    const buf = readFileSync(join(STATIC_MODEL.dir, 'model.safetensors'));
    const headerLen = Number(buf.readBigUInt64LE(0));
    const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString('utf8')) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
    const key = Object.keys(header).find((k) => k !== '__metadata__')!;
    const { dtype, shape, data_offsets } = header[key];
    if (dtype !== 'F32') throw new Error(`${STATIC_MODEL.id}: expected F32 weights, got ${dtype}`);
    // A view into the file buffer when the data is 4-byte aligned (safetensors pads its header to 8), so
    // the 29 MB table is held once; a copy only otherwise.
    const start = buf.byteOffset + 8 + headerLen + data_offsets[0], bytes = data_offsets[1] - data_offsets[0];
    const weights = start % 4 === 0 ? new Float32Array(buf.buffer, start, bytes / 4) : new Float32Array(buf.buffer.slice(start, start + bytes));
    return createStaticEmbedder({ modelId: STATIC_MODEL.id, tokenizerBytes: new Uint8Array(readFileSync(join(STATIC_MODEL.dir, 'tokenizer.json'))), weights, dimensions: shape[1] });
  })();
  return loaded;
}
