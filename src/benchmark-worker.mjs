// Fits one-vs-rest logistic heads for a share of the labels (see benchmark.ts). Plain JavaScript:
// Node 20 doesn't apply tsx's loader to worker threads.
import { parentPort } from 'node:worker_threads';

import { fitLogistic } from '@liquidau/solvers';

parentPort.on('message', (job) => {
  const models = job.targets.map((target) => {
    const m = fitLogistic(job.X, job.labels.map((l) => (l === target ? 1 : 0)), { C: 1 });
    return { target, coef: m.coef, intercept: m.intercept };
  });
  parentPort.postMessage(models);
});
