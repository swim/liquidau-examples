/**
 * The release gates every pipeline here applies (the walkthrough, the release pipeline, and the lab's
 * research rounds, which import this file through a symlink): one gate set, so a result found in the lab
 * is judged by the same gates a release is.
 *
 *   calibrationAlpha   the calibration test's level: probabilities are refused as miscalibrated at
 *                      p <= 0.01 (trainHeads' calibrationAlpha). The test is the library default, Cox's
 *                      test with its level corrected for the calibrator's own error ('cox-corrected');
 *                      it checks the logistic recalibration, not the curve's shape
 *   sliceGate          a recall head with slices must hold its recall in every slice with at least
 *                      this many positives (HeadPolicy.sliceGate)
 */
export const GATES = { calibrationAlpha: 0.01 } as const;
export const SLICE_GATE = { minPositives: 10 } as const;
