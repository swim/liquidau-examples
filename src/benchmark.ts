/**
 * A published benchmark, reproduced with this stack: MTEB's Banking77Classification.
 *
 * Protocol (mteb 1.12.75, mteb/abstasks/AbsTaskClassification.py): 8 training examples per intent
 * from Banking77's train file, logistic regression (C = 1) on the frozen embeddings, accuracy on the
 * 3,080-message test file, averaged over 10 experiments with different samples. Published results
 * (github.com/embeddings-benchmark/results): all-MiniLM-L6-v2 80.04, all-mpnet-base-v2 81.70.
 *
 * Differences, so a gap of about a point is expected: embeddings come from quantised (q8) ONNX
 * models via transformers.js rather than PyTorch; the classifier is 77 one-vs-rest heads fitted with
 * solvers' fitLogistic (exact Newton) and the highest score wins, where scikit-learn fits one
 * multinomial model with L-BFGS capped at 100 iterations; the 8 examples per intent are a different
 * random draw. What it checks: that the embedding pipeline and solvers' logistic regression give
 * published-quality features and fits.
 */
import { availableParallelism } from 'node:os';
import { Worker } from 'node:worker_threads';

import { decisionFunction, seededRandom } from '@liquidau/solvers';

import { loadBanking77 } from './data.ts';
import { makeEmbedder, MODEL, SMALL_MODEL } from './embed.ts';

const PUBLISHED: Record<string, number> = { [SMALL_MODEL]: 80.04, [MODEL]: 81.7 };
const EXPERIMENTS = 10, PER_LABEL = 8;

const messages = await loadBanking77();
const intents = [...new Set(messages.map((m) => m.intent))].sort();
const label = new Map(intents.map((x, i) => [x, i]));
const train = messages.filter((m) => m.split === 'train'), test = messages.filter((m) => m.split === 'test');
console.log(`Banking77: ${train.length} train, ${test.length} test, ${intents.length} intents; ${EXPERIMENTS} experiments × ${PER_LABEL} examples per intent`);

const workers = Array.from({ length: Math.max(1, Math.min(availableParallelism() - 1, 12)) }, () =>
  new Worker(new URL('./benchmark-worker.mjs', import.meta.url)));
const fitAll = async (X: number[][], labels: number[]) => {
  const shares = workers.map((_, w) => intents.map((_, i) => i).filter((i) => i % workers.length === w));
  const parts = await Promise.all(workers.map((worker, w) => new Promise<Array<{ target: number; coef: number[]; intercept: number }>>((resolve) => {
    worker.once('message', resolve);
    worker.postMessage({ X, labels, targets: shares[w] });
  })));
  return parts.flat().sort((a, b) => a.target - b.target);
};

for (const model of [SMALL_MODEL, MODEL]) {
  const embedder = await makeEmbedder(() => {}, model);
  const vec = new Map((await embedder.embedMany(messages.map((m) => m.text))).map((v, i) => [messages[i].id, v]));
  const accuracies: number[] = [];
  const t0 = Date.now();
  for (let e = 0; e < EXPERIMENTS; e++) {
    // Undersample: shuffle the train file, keep the first PER_LABEL of each intent.
    const rand = seededRandom(42 + e);
    const order = [...train];
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    const counts = new Map<string, number>();
    const picked = order.filter((m) => { const c = counts.get(m.intent) ?? 0; if (c >= PER_LABEL) return false; counts.set(m.intent, c + 1); return true; });
    const models = await fitAll(picked.map((m) => vec.get(m.id)!), picked.map((m) => label.get(m.intent)!));
    let correct = 0;
    for (const m of test) {
      const x = vec.get(m.id)!;
      let best = -1, bestScore = -Infinity;
      for (const h of models) { const s = decisionFunction(h, x); if (s > bestScore) { bestScore = s; best = h.target; } }
      if (best === label.get(m.intent)) correct++;
    }
    accuracies.push(correct / test.length);
    process.stdout.write(`\r${model}: experiment ${e + 1}/${EXPERIMENTS}`);
  }
  const mean = accuracies.reduce((a, b) => a + b, 0) / accuracies.length;
  const sd = Math.sqrt(accuracies.reduce((a, b) => a + (b - mean) ** 2, 0) / (accuracies.length - 1));
  console.log(`\r${model}: accuracy ${(100 * mean).toFixed(2)} ± ${(100 * sd).toFixed(2)} (published ${PUBLISHED[model].toFixed(2)}; difference ${(100 * mean - PUBLISHED[model]).toFixed(2)}) in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
await Promise.all(workers.map((w) => w.terminate()));
