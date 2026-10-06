# liquidau-examples

A worked example of the three `@liquidau` libraries together, on real customer-support messages:

- **[solvers](https://www.npmjs.com/package/@liquidau/solvers)** - the statistics: model fitting, and how sure a labelled sample lets you be.
- **[rule-miner](https://www.npmjs.com/package/@liquidau/rule-miner)** - readable rules, mined from labelled examples and certified before they decide anything.
- **[embedding-classifier](https://www.npmjs.com/package/@liquidau/embedding-classifier)** - calibrated heads on text embeddings, each with a stated guarantee, shipped as a checked artifact.

It covers the whole loop: mining and certifying rules, sampling and labelling evaluation sets,
training heads with stated guarantees, shipping, planning the next labelling round, monitoring for
drift, and serving with rules in front of the model.

## Contents

| | What | Run |
|---|---|---|
| [Walkthrough](#walkthrough-banking77) | Every stage on Banking77, end to end | `npm start` |
| [Serving](#serving) | A serverless-shaped handler, checked against always embedding | `npm run serve:check` |
| [Release pipeline](#release-pipeline) | From a traffic sample to a paired, validated release, with an acceptance test | `npm run release:simulate` |
| [ToxicChat](#second-dataset-toxicchat) | The same loop on chatbot prompts | `npm run toxicchat` |
| [Benchmark](#benchmark-mteb-banking77classification) | A published benchmark reproduced with this stack | `npm run benchmark` |

## Setup

```bash
npm install
```

The libraries are pinned to exact published versions (`@liquidau/solvers` 0.6.2,
`@liquidau/rule-miner` 0.5.0, `@liquidau/embedding-classifier` 0.6.2), so the results below
reproduce. See [Library versions](#library-versions) to try others.

## Walkthrough: Banking77

[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) contains 13,083 online-banking
customer queries labelled with 77 intents, published by PolyAI under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Casanueva et al., *Efficient Intent
Detection with Dual Sentence Encoders*, 2020). `npm start` downloads it into `data/`; it is not
redistributed here.

The example defines two heads over the intents:

| Head | Intents | Policy |
|---|---|---|
| `urgent` | compromised card, lost or stolen card or phone, card payment, cash withdrawal or direct debit not recognised | Recall: a miss is costly |
| `refund` | request refund, refund not showing up | Precision: a false alarm is costly |

Banking77 is roughly balanced across intents, so `urgent` makes up about 8% of it, far more than real
traffic. The scenario (`src/data.ts`) builds a realistic stream instead:

- **Traffic:** `urgent` at 4% and `refund` at 2%.
- **Historical labelled set:** the subsampled-away positives, plus some negatives, form an enriched
  set of past escalations. It is training data only.
- **Splits by group:** the traffic is split into an unlabelled veto background (for rule mining), a
  certify background (for rule certification), a budget background (for thresholds and as the drift
  reference), a live window, and the sampling frame. Exact duplicate texts share a group, so no group
  straddles two roles.

```bash
npm start   # first run about 4 minutes (two embedding models), then about 1 minute; embeddings are cached in .cache/
```

Embeddings come from `Xenova/all-mpnet-base-v2`, run locally through `@huggingface/transformers`, so
no API key is needed; the smaller `all-MiniLM-L6-v2` is used for comparison. The first run downloads
both models, about 130 MB together. Outputs go to `out/`: `artifact.json` (the shipped classifier),
`ruleset.json` (the certified rules), `report.md` (gates, design-based estimates, guarantees and
coverage) and `comparison.md` (baselines and error analysis).

| Stage | Library | What it demonstrates |
|---|---|---|
| 1 rules | rule-miner | `mineRules` with a veto background given as records (P2 and P3 checked) and `validateRules`. `certifyPrefixes` on certify records shows why firing-rate certification is the wrong statement at real prevalence. Plus hard-negative briefs |
| 2 sample | embedding-classifier, solvers | A round-0 model scores the frame from text only. `designSample` stratifies by rule firing × score band and merges small strata. A pilot round gives per-stratum rates for both heads, and `requiredSampleSize` sizes the main sample with `expected-positives` |
| 3 label | embedding-classifier, rule-miner | `retrieveFromSeeds` finds candidates near historical positives. `labelQueue` blinds reviewers (they see only an id, the text and the heads); a simulated reviewer answers from the gold labels and skips 2%. `applyReviews` recomputes inclusion probabilities. `certifyFalseAlarms` then certifies the rules on the calibration sample's negatives, capped by their known firing on the frame, and `evaluateFired` measures them on test with design weights |
| 4 train | embedding-classifier | `trainHeads` with records: provenance rules P1–P7, P6 caps on retrieved positives, design-weighted calibration, `mode: 'design'` thresholds for a recall head and a precision head, the rules as baseline, a per-slice gate, and the Cox calibration gate |
| 5 ship | embedding-classifier | `assertRoundTrip`, `reportMarkdown` with a coverage section, `publishPlan` and `decide` on example messages |
| 6 next round | rule-miner | `disagreementQueues`: what to label next, for rules and classifier alike |
| 7 monitor | embedding-classifier | `monitorWindow` on a normal live window, then on one with a fraud wave injected |
| 8 compare | all three | The same split and test records against baselines: the certified rules alone, a heuristic threshold, and a smaller embedding model (`all-MiniLM-L6-v2`). Plus `urgent`'s false alarms and misses by Banking77 intent. Written to `out/comparison.md` |

## Serving

`src/serve.ts` is a serving handler shaped for serverless functions:
- The artifact and rule set are loaded once per instance.
- Each request runs rules, then the embedding, the heads and `decide`.
- It imports only the package roots, which use no Node built-ins, so it also runs on edge runtimes
  when `embed` calls a hosted API.

Under a "rules or classifier" policy, a rule hit on the top-priority, unsuppressable head decides
without an embedding. A 5% sample of those messages is still embedded, so drift windows see all
traffic.

`npm run serve:check` (after `npm start`) compares the handler with always embedding on all 13,083
Banking77 messages, under three policies:
- 0 mismatches
- 5% of messages skip the embedding where the policy allows it, and none where it doesn't
- about 16 µs per message outside the embedding

**Rules in front of the model.** The rules can also run in their own small function (or at the edge)
and pass only the messages they can't settle to the model's function. embedding-classifier's
`settleWithRules` decides from the rules alone exactly when that is what the model tier would decide,
and `checkRuleSetPairing` refuses a model tier certified with a different rule set. The design and the
embedding options are in embedding-classifier's [docs/SERVING.md](https://github.com/swim/embedding-classifier/blob/main/docs/SERVING.md).

## Release pipeline

`src/release.ts` is a reference training pipeline, from a sample of real traffic to a release you can
deploy:

1. **`plan`** - embed the traffic, score it (with an independent offline teacher when one is
   available), draw a stratified probability sample and write one blinded labelling queue.
2. **`train`** - apply the reviewers' labels, mine and certify firing rules and joint dismissal rules,
   train each head (`type: 'auto'` picks linear, kNN or a stack per label; thresholds carry the
   strongest guarantee the data supports; optional caps on how often a head may fire), then build and
   validate the artifact.
3. **Release** - `ruleset.json`, `artifact.json`, `manifest.json` (pairing the two by hash, with the
   embedder's spec) and `report.md`. `loadRelease` reads it back the way a runtime should: validated,
   paired, and refused if the runtime's embedder differs.

`src/release-simulate.ts` runs the pipeline on a benchmark with a simulated reviewer, then checks every
promise on the benchmark's fully labelled future file, through the two tiers:

```bash
npm run export:onnx                        # once: the multi-layer embedding model (needs Python with torch, transformers, onnx, onnxruntime)
npm run release:simulate -- abcd           # or ade, toxicchat
```

The pipeline embeds with `src/embed-layers.ts`: all-mpnet-base-v2 layers 4, 8 and 12, mean-pooled and
concatenated, through the ONNX export. It truncates exactly as the training library does, so training
and serving features match. Acceptance results (`out/release-sim/<dataset>/acceptance.md`, seed 7):

| Release | Head | Promise | Met on the future file | Recall | False alarms |
|---|---|---|---|---|---|
| ABCD support chats | account access | recall ≥ 0.85 | yes | 0.886 | 2.55% |
| | refund | precision ≥ 0.80 | yes (0.830) | 0.765 | 0.84% |
| ADE (adverse drug events) | adverse event | recall ≥ 0.85 | yes | 0.910 | 22.2% |
| ToxicChat | toxic | recall ≥ 0.80 | yes | 0.824 | 5.51% |
| | jailbreak | refused: too few labelled positives for a promise within its 8% firing cap | - | 0.840 | 5.71% |

Datasets for the simulation, downloaded into `data/` (not redistributed):
- **ABCD** - [Action-Based Conversations Dataset](https://github.com/asappresearch/abcd) (MIT): `data/abcd/abcd_v1.1.json.gz`.
- **ADE** - [ADE Corpus V2](https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2), classification config, converted to `data/ade/classification.json`.
- **ToxicChat** - downloaded at run time (see below).

## Second dataset: ToxicChat

`npm run toxicchat:embed` (about 30 minutes, once) then `npm run toxicchat`. **The data contains
offensive text.** [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) (LMSYS, CC BY-NC 4.0,
version 0124) is 10,165 real prompts to a chatbot, labelled toxic (7.3%) and jailbreak (2.0%). It
is downloaded at run time, not shipped.

The train file is the traffic: a first labelled round, background for rules and budgets, and a frame
for the stratified sample. The test file is a later period with every label known, so the guarantees
are checked against the truth rather than a sample. Results (`out/toxicchat.md`):

- **Benchmark:** average precision 0.795 for `toxic`, between the dataset's published
  ToxicChat-T5-large (0.885, fine-tuned) and OpenAI Moderation (0.631).
- **`toxic`:** stated recall ≥ 0.8 (design, linearised) and reached 0.766 (0.717–0.808) on the test
  file, with precision 0.59. The guarantee is about traffic like the sampled population; the test
  file is a later period, and a shift between them can move recall below the target.
- **`jailbreak`:** with about 10 calibration positives, the head fails the Cox calibration gate.
- **Rules:** 3 certified, catching 5.5% of toxic prompts at 72% precision. Toxicity is less lexical
  than banking intents.
- **Monitor:** quiet on a random production window. The test file is ordered: its first 1,200 rows
  are 19% toxic, and that window raised an alert.

### Teacher labels

`teacher-labels/` holds an offline teacher's answers (Clef on Cloudflare Workers AI) for ToxicChat
and Banking77, used to choose and stratify what people label. See `manifest.json` there.
- **Training-only:** they never calibrate, set thresholds or certify.
- **No prompt text:** each entry is keyed by a hash of the model, the questions and the prompt.
- **Reruns cost nothing:** `npm run teacher:toxicchat -- [--model clef]` sends only prompts that are
  missing. It needs Cloudflare credentials in `~/.config/liquidau/cloudflare.env`.

## Benchmark: MTEB Banking77Classification

`npm run benchmark` (about 20 minutes, spread across CPU cores) reproduces a published benchmark
with this stack: [MTEB](https://github.com/embeddings-benchmark/mteb)'s Banking77Classification.
- **Protocol** (mteb 1.12.75): 8 training examples per intent from Banking77's train file, logistic
  regression (C = 1) on frozen embeddings, accuracy on the 3,080-message test file, averaged over 10
  runs.
- **Published results** ([embeddings-benchmark/results](https://github.com/embeddings-benchmark/results)):
  `all-MiniLM-L6-v2` 80.04, `all-mpnet-base-v2` 81.70.

It checks that the local embedding pipeline and solvers' logistic regression produce
published-quality features and fits. A gap of about a point is expected:
- **Embeddings:** quantised (q8) ONNX models rather than PyTorch.
- **Classifier:** 77 one-vs-rest heads fitted with solvers' exact Newton solver, where scikit-learn
  fits one multinomial model with L-BFGS.
- **Training draw:** a different random sample of the 8 examples per intent.

| Model | This stack | Published | Difference |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 79.19 ± 1.27 | 80.04 | −0.85 |
| `all-mpnet-base-v2` | 80.03 ± 0.91 | 81.70 | −1.67 |

MiniLM is within run-to-run variation. mpnet is about 1.7 points low, more than expected; 8-bit
quantisation affects the larger model more. The ranking matches the published one.

## Library versions

The `@liquidau` dependencies are pinned to exact versions in `package.json`. To try another
version:

1. Install it exactly, for example `npm install --save-exact @liquidau/embedding-classifier@<version>`.
2. Run `npm start`.
3. Compare the console output and `out/report.md` with the previous run. Everything is seeded, so
   differences come from the version change.

## Licence

The code is [MIT](LICENSE). The datasets keep their own licences and are downloaded at run time, not
redistributed. The teacher labels in `teacher-labels/` are derived from them and are not covered by
the MIT licence: the `clef-toxicchat-*` files follow ToxicChat's CC BY-NC 4.0 (non-commercial use
only), and `clef-banking77-clef.json` follows Banking77's CC BY 4.0.
