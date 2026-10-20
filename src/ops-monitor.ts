/**
 * Operations on router telemetry. The drift checks and the retraining trigger live in @liquidau/router
 * (`windowOf`, `checkWindow`; see its README): scores per head in two strata, the rules' and the edge's
 * decision rates, and unavailable results, under one family-wise false-alarm budget. This file adds a
 * prevalence estimate for each head (BBSE, below), which stays in the examples: it is only valid under
 * label shift, and on ToxicChat's real shift its interval covered the truth 2 times in 20 in the lab.
 */
import { checkWindow, windowOf, type TelemetryWindow, type WindowCheck, type WindowReport } from '@liquidau/router';

export { checkWindow, windowOf };
export type { TelemetryWindow, WindowReport };
/** The router's name is WindowCheck; kept as Check for the examples' callers. */
export type Check = WindowCheck;

/** A head's routing on the release's labelled test split: routed positives and routed negatives. */
export interface HeadConfusion { truePositives: number; positives: number; falsePositives: number; negatives: number }

export interface PrevalenceEstimate {
  /** Share of available requests routed to the head. */
  routedRate: number;
  /** Estimated share of requests that are positive for the head, with a 95% interval. */
  prevalence: number;
  lower: number;
  upper: number;
}

/**
 * Each head's prevalence in a window, from its routing rate alone (black-box shift estimation, BBSE:
 * Lipton et al. 2018): pi = (q - FPR) / (TPR - FPR), where q is the window's routed rate and TPR / FPR
 * the release's routing rates on its labelled test split. The 95% interval is the delta method over the
 * window's and the test split's binomial errors. It assumes label shift - positives and negatives each
 * look as they did at release, only their mix changed - and is not validated when the messages
 * themselves change (lab, Stage MC-T3). The test rates are unweighted, which matches the design-weighted
 * rates when the sample is drawn with equal inclusion probabilities (ABCD's proportional design).
 */
export function estimatePrevalence(live: TelemetryWindow, confusion: Readonly<Record<string, HeadConfusion>>): Record<string, PrevalenceEstimate> {
  const n = live.requests - live.unavailable;
  const out: Record<string, PrevalenceEstimate> = {};
  for (const [head, c] of Object.entries(confusion)) {
    const q = (live.routed[head] ?? 0) / n, tpr = c.truePositives / c.positives, fpr = c.falsePositives / c.negatives, d = tpr - fpr;
    if (!(n > 0) || !(d > 0)) continue;
    const pi = Math.min(1, Math.max(0, (q - fpr) / d));
    const v = (q * (1 - q)) / n + ((1 - pi) ** 2 * fpr * (1 - fpr)) / c.negatives + (pi ** 2 * tpr * (1 - tpr)) / c.positives;
    const half = (1.96 * Math.sqrt(v)) / d;
    out[head] = { routedRate: q, prevalence: pi, lower: Math.max(0, pi - half), upper: Math.min(1, pi + half) };
  }
  return out;
}
