/**
 * Labels every ToxicChat prompt (English, train and test files) with Clef as an offline teacher, then
 * reports the teacher's own ranking quality on the test file next to our head and the published
 * fine-tuned T5. The labels are training-only (to distil, or to choose and stratify what people
 * label) and never calibrate or certify anything.
 *
 * The questions paraphrase ToxicChat's own label definitions and were fixed before seeing any
 * result; tuning them on test labels would leak.
 *
 * Usage: toxicchat-clef.ts [--dry-run] [--model clef-flash|clef]
 */
import { join } from 'node:path';

import { decisionFunction, fitLogistic } from '@liquidau/solvers';

import { makeEmbedder } from '../embed.ts';
import { averagePrecision } from '../metrics.ts';
import { loadToxicChat, MAX_CHARS } from '../toxicchat.ts';
import { ClefTeacher, type ClefModel } from './clef.ts';
import { TOXICCHAT_QUESTIONS } from './questions.ts';

const args = process.argv.slice(2);
const model = (args[args.indexOf('--model') + 1] === 'clef' ? 'clef' : 'clef-flash') as ClefModel;
// Tracked in the repository (teacher-labels/, with a manifest): small, and they hold no prompt text.
export const teacherCachePath = (m: ClefModel) => join(import.meta.dirname, '..', '..', 'teacher-labels', `clef-toxicchat-${m}.json`);

if (import.meta.url === `file://${process.argv[1]}`) {
  const prompts = await loadToxicChat();
  const texts = prompts.map((p) => p.text);
  if (args.includes('--dry-run')) {
    // About 4 characters per token, plus about 330 tokens of questions and framing per request
    // (measured: 1,001 tokens for 3 short prompts).
    const qTokens = 330;
    const tokens = texts.reduce((a, t) => a + Math.ceil(Math.min(t.length, 8000) / 4) + qTokens, 0);
    console.log(`${texts.length} prompts, about ${tokens.toLocaleString()} input tokens; at list price about $${((tokens / 1e6) * (model === 'clef' ? 0.24 : 0.09)).toFixed(2)} with ${model}`);
    process.exit(0);
  }
  const teacher = new ClefTeacher({ model, questions: TOXICCHAT_QUESTIONS, cachePath: teacherCachePath(model), maxTokens: 4_500_000, log: (l) => console.error(l) });
  const answers = await teacher.label(texts);
  // The teacher's own ranking quality on the test file, against our head on the official split.
  const vectors = await (await makeEmbedder(() => {}, undefined, { cache: 'toxicchat', maxChars: MAX_CHARS })).embedMany(texts);
  const train = prompts.map((_, i) => i).filter((i) => prompts[i].split === 'train'), test = prompts.map((_, i) => i).filter((i) => prompts[i].split === 'test');
  for (const h of ['toxic', 'jailbreak'] as const) {
    const y = test.map((i) => prompts[i].truth[h]);
    const head = fitLogistic(train.map((i) => vectors[i]), train.map((i) => prompts[i].truth[h]), { C: 1, classWeight: 'balanced' });
    const ours = averagePrecision(test.map((i) => decisionFunction(head, vectors[i])), y);
    const clef = averagePrecision(test.map((i) => answers[i][h]), y);
    const agree = train.filter((i) => (answers[i][h] >= 0.5 ? 1 : 0) === prompts[i].truth[h]).length / train.length;
    console.log(`${h}: AP on the test file - Clef (${model}, zero-shot) ${clef.toFixed(3)}, our mpnet head ${ours.toFixed(3)}${h === 'toxic' ? ', published ToxicChat-T5-large 0.885' : ''}; Clef agrees with the train labels at p >= 0.5 on ${(100 * agree).toFixed(1)}%`);
  }
}
