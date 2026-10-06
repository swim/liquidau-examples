/** Average precision of `scores` against binary labels (ties share a step), as scikit-learn computes it. */
export function averagePrecision(scores: readonly number[], y: readonly (0 | 1)[]): number {
  const order = scores.map((s, i) => [s, y[i]] as const).sort((a, b) => b[0] - a[0]);
  const positives = y.reduce<number>((a, b) => a + b, 0);
  let tp = 0, fp = 0, ap = 0, prevRecall = 0;
  for (let i = 0; i < order.length; i++) {
    if (order[i][1]) tp++; else fp++;
    if (i + 1 < order.length && order[i + 1][0] === order[i][0]) continue;
    const recall = tp / positives;
    ap += (recall - prevRecall) * (tp / (tp + fp));
    prevRecall = recall;
  }
  return ap;
}
