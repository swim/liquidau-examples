/**
 * EmbeddingGemma (google/embeddinggemma-300m, the onnx-community 8-bit export), the examples' default
 * encoder. The vectors are the model's `sentence_embedding` output (mean pooling and the model's dense
 * projection, not a mean over hidden states) for the prompt "task: classification | query: ", unit
 * length: exactly as liquidau-lab measured it (stages EG, JC, SL, DY).
 *
 * Truncation is declared: the text is cut to `maxChars` (surrogate-safe), the prompt prepended, then
 * transformers.js truncates to 512 tokens. EmbeddingGemma is a Gemma model: using it means accepting
 * the Gemma Terms of Use (https://ai.google.dev/gemma/terms).
 */
import { AutoModel, AutoTokenizer } from '@huggingface/transformers';
import { truncateText, type EmbeddingSpec } from '@liquidau/embedding-classifier';

export const GEMMA_MODEL = 'onnx-community/embeddinggemma-300m-ONNX';
export const GEMMA_PROMPT = 'task: classification | query: ';
export const GEMMA_DIMENSIONS = 768;
export const GEMMA_MAX_TOKENS = 512;
const DEFAULT_MAX_CHARS = 2000;

/** The spec an artifact records: the prompt is the input type, part of what the vectors mean. */
export const gemmaSpec = (maxChars = DEFAULT_MAX_CHARS): EmbeddingSpec => ({
  model_id: GEMMA_MODEL, dimensions: GEMMA_DIMENSIONS, normalize: true, precision: 'q8', input_type: GEMMA_PROMPT, max_chars: maxChars,
});

export interface GemmaEmbedder {
  spec: EmbeddingSpec;
  embed(texts: readonly string[]): Promise<number[][]>;
}

export async function makeGemmaEmbedder(options: { maxChars?: number } = {}): Promise<GemmaEmbedder> {
  const maxChars = options.maxChars ?? DEFAULT_MAX_CHARS;
  const tokenizer = await AutoTokenizer.from_pretrained(GEMMA_MODEL);
  const model = await AutoModel.from_pretrained(GEMMA_MODEL, { dtype: 'q8' });
  return {
    spec: gemmaSpec(maxChars),
    async embed(texts) {
      if (!texts.length) return [];
      const inputs = await tokenizer(texts.map((t) => GEMMA_PROMPT + truncateText(t, maxChars)), { padding: true, truncation: true, max_length: GEMMA_MAX_TOKENS });
      const rows = (await model(inputs)).sentence_embedding.tolist() as number[][];
      return rows.map((v) => { const n = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1; return v.map((x) => x / n); });
    },
  };
}
