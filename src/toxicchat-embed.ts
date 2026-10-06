/**
 * Embeds every ToxicChat prompt once (cached in .cache/embeddings-toxicchat-<model>.json).
 * Usage: toxicchat-embed.ts [model id]; defaults to the shipped model.
 */
import { MODEL, makeEmbedder } from './embed.ts';
import { loadToxicChat, MAX_CHARS } from './toxicchat.ts';

const model = process.argv[2] ?? MODEL;
const prompts = await loadToxicChat();
const embedder = await makeEmbedder((line) => console.log(line), model, { cache: 'toxicchat', maxChars: MAX_CHARS });
const t0 = Date.now();
await embedder.embedMany(prompts.map((p) => p.text));
console.log(`embedded ${prompts.length} prompts with ${model} in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
