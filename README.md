# liquidau-examples

A worked example of the `@liquidau` libraries together, on real customer-support messages:

- **[solvers](https://www.npmjs.com/package/@liquidau/solvers)** - the statistics: model fitting, and how sure a labelled sample lets you be.
- **[rule-miner](https://www.npmjs.com/package/@liquidau/rule-miner)** - readable rules, mined from labelled examples and certified before they decide anything.
- **[embedding-classifier](https://www.npmjs.com/package/@liquidau/embedding-classifier)** - calibrated heads on text embeddings, each with a stated guarantee, shipped as a checked artifact.
- **[router](https://www.npmjs.com/package/@liquidau/router)** - serves a complete release (classifier, rules, policy and evaluation evidence, pinned by one manifest digest): rules first, the embedding only when a score could change the outcome.
- **[text-preprocessing](https://www.npmjs.com/package/@liquidau/text-preprocessing)** - shared identities and checks (encoder identity, canonical digests) the others build on.

It covers the whole loop: mining and certifying rules, sampling and labelling evaluation sets,
training heads with stated guarantees, shipping, planning the next labelling round, monitoring for
drift, and serving with rules in front of the model.

## Contents

| | What | Run |
|---|---|---|
| [Walkthrough](#walkthrough-banking77) | Every stage on Banking77, end to end | `npm start` |
| [Serving](#serving) | A serverless-shaped handler on the router, checked against always embedding | `npm run serve:check` |
| [Split deployment](#split-deployment-rules-function-and-model-function) | A rules function in front of a model function (Lambda and Worker wrappers), checked against one router; cold starts and latency | `npm run split:check`, `npm run serve:bench` |
| [Operations](#operations) | Drift checks on router telemetry, a retraining trigger, and the promote and rollback runbook | `npm run ops:check` |
| [Release pipeline](#release-pipeline) | From a traffic sample to a router release, with an acceptance test that routes a later period | `npm run release:simulate` |
| [ToxicChat](#second-dataset-toxicchat) | The same loop on chatbot prompts | `npm run toxicchat` |
| [Benchmark](#benchmark-mteb-banking77classification) | A published benchmark reproduced with this stack | `npm run benchmark` |
| [Showcase page](#showcase-page) | A routing-pathways diagram and a card per dataset, generated from the releases | `npm run site` |

## Setup

```bash
npm install
```

> **The examples are ahead of the published libraries.** They use features that are not released yet
> (the edge tier, the middle tier, the finer-label score, context features, and rule-miner's `firing`).
> On the pinned versions below, the scripts fail at startup with a missing export (for example
> `createEdgeTier` or `createStaticEmbedder`). Until the next releases, run them against local
> checkouts of the libraries. Link the five repos into `node_modules/@liquidau` (as liquidau-lab's
> `npm run link-modules` does) and run with `node --preserve-symlinks`. The numbers in this README
> come from runs on those local versions.

The libraries are pinned to exact published versions (`@liquidau/solvers` 0.6.2,
`@liquidau/rule-miner` 0.5.2, `@liquidau/embedding-classifier` 0.8.1, `@liquidau/router` 0.2.2,
`@liquidau/text-preprocessing` 0.1.0). Once the libraries are released, bump these pins and the
results below reproduce on them. See [Library versions](#library-versions) to try others.

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
npm start   # the first run embeds with two models and takes longer; then about 1 minute (embeddings are cached in .cache/)
```

Embeddings come from EmbeddingGemma (`onnx-community/embeddinggemma-300m-ONNX`, 8-bit, with its
classification prompt), run locally through `@huggingface/transformers`, so no API key is needed. The
previous default, `Xenova/all-mpnet-base-v2`, is used for comparison. The first run downloads both
models, about 440 MB together. Outputs go to `out/`: `artifact.json` (the shipped classifier),
`ruleset.json` (the certified rules), `report.md` (gates, design-based estimates, guarantees and
coverage) and `comparison.md` (baselines and error analysis).

| Stage | Library | What it demonstrates |
|---|---|---|
| 1 rules | rule-miner | `mineRules` with a veto background given as records (P2 and P3 checked) and `validateRules`. `certifyPrefixes` on certify records shows why firing-rate certification is the wrong statement at real prevalence. Plus hard-negative briefs |
| 2 sample | embedding-classifier, solvers | A round-0 model scores the frame from text only. `designSample` stratifies by rule firing × score band and merges small strata. A pilot round gives per-stratum rates for both heads, and `requiredSampleSize` sizes the main sample with `expected-positives` |
| 3 label | embedding-classifier, rule-miner | `retrieveFromSeeds` finds candidates near historical positives. `labelQueue` blinds reviewers (they see only an id, the text and the heads); a simulated reviewer answers from the gold labels and skips 2%. `applyReviews` recomputes inclusion probabilities. `certifyFalseAlarms` then certifies the rules on the calibration sample's negatives, capped by their known firing on the frame, and `evaluateFired` measures them on test with design weights |
| 4 train | embedding-classifier | `trainHeads` with records: provenance rules P1–P7, P6 caps on retrieved positives, design-weighted calibration, `mode: 'design'` thresholds for a recall head and a precision head, the rules as baseline, a per-slice gate, and the calibration gate (Cox's test, corrected for the calibrator's own error) |
| 5 ship | embedding-classifier | `assertRoundTrip`, `reportMarkdown` with a coverage section, `publishPlan` and `decide` on example messages |
| 6 next round | rule-miner | `disagreementQueues`: what to label next, for rules and classifier alike |
| 7 monitor | embedding-classifier | `monitorWindow` on a normal live window, then on one with a fraud wave injected |
| 8 compare | all three | The same split and test records against baselines: the certified rules alone, a heuristic threshold, and the previous encoder (`all-mpnet-base-v2`). Plus `urgent`'s false alarms and misses by Banking77 intent. Written to `out/comparison.md` |

On the same test records, `urgent` reaches recall 0.931 (certified ≥ 0.856) with 0.2% false alarms and
precision 0.95; mpnet at the same guarantee reaches 0.977 with 1.5% false alarms and precision 0.74.
`refund` is at precision 1.000 and recall 1.000 on EmbeddingGemma (0.977 recall on mpnet). Every gate
passes (16 October, on the libraries in this workspace).

**The finer-label route score.** `FINER=1 npm start` labels each message's intent as well (the finer
label under `urgent` and `refund`) and trains both heads with embedding-classifier's `finer` option. A
multinomial model over the 77 intents is fitted on the training records, and each head scores the
embedding plus Σ P(intent) × the intent's route rate. Outputs go to `out/finer/`. Measured on mpnet,
before the switch to EmbeddingGemma, `urgent` went from 1.1% to 0.2% false alarms (certified bound 1.6%
to 0.56%) and from 0.79
to 0.94 precision, with recall 0.924 to 0.900, still guaranteed at 0.85. It needs an
embedding-classifier release with `finer` (newer than the pinned version). The finer model is six
multinomial fits (the model and five folds, split by paraphrase group) over 77 intents and 3,456
training rows at 768 dimensions; the whole `FINER=1` run took 14 minutes here with cached embeddings.
Every fit converges and the gates pass (rerun 14 October).

## Serving

`npm start` also writes a complete router release to `out/release/`: the classifier, its rule set,
the routing policy and evaluation evidence from the held-out test records, pinned by `manifest.json`'s
digest (`deploy.json` holds the digest a deployment would pin). `src/serve.ts` is a serving handler on
it, shaped for serverless functions:
- `loadRouter` validates the release once per instance: every file's digest, the rule set's pairing
  with the classifier, the encoder's exact identity and, in enforce mode, the release's gates and
  evidence.
- Each request runs the rules on the original text; when no score could change the outcome it is
  settled without an embedding. Otherwise the router embeds, scores and applies the policy.
- A deterministic 5% of rules-settled requests is still embedded for monitoring, after the result is
  returned; the handler awaits `router.flush()` so a function finishes that work before it returns.
- It imports only the router's package root, which uses no Node built-ins: the host supplies the
  release source and the encoder (`src/router-host.ts` has a filesystem source and encoder adapters).

`npm run serve:check` (after `npm start`) serves releases with three policies and compares every
decision with the always-embed reference on all 13,083 Banking77 messages:
- 0 mismatches under every policy
- 5.0% of messages are settled without the embedding where the policy allows it (`urgent` first), and
  none where a rule's head could be outranked or suppressed
- about 20-26 µs per message for the router itself (embeddings cached)

**Rules in front of the model.** The rules can also run in their own small function (or at the edge)
and pass only the messages they can't settle to the model's function; the design and the embedding
options are in embedding-classifier's [docs/SERVING.md](https://github.com/swim/embedding-classifier/blob/main/docs/SERVING.md).

## Split deployment: rules function and model function

`src/hosts.ts` deploys a release as two functions: a small **rules function** (`loadRulesTier`: the
manifest, rules, edge tier, policy and evidence, with the small static encoder the edge uses - no main
encoder, no classifier) in front of a **model function**
(`loadRouter`, sized for the encoder). Both pin the same manifest digest. The rules function answers
what the rules settle and forwards the rest as JSON; a request it answers and selects for monitoring is
sent on after the answer, for telemetry only. `lambdaHandler` and `workerHandler` wrap either function
for AWS Lambda (function URL or API Gateway events) or a Cloudflare Worker (`ctx.waitUntil` for the
work after the answer); `httpSender` connects the rules function to the model function's URL.

`npm run split:check` (after `npm start` and the release simulations) routes every message of every
release both ways - through one router, and rules function -> JSON -> model function - and requires
identical results and identical telemetry:

| Release | Mode | Messages | Answered by the rules function | Forwarded to the model function | Monitoring forwards | Mismatches | Telemetry identical |
|---|---|---|---|---|---|---|---|
| Banking77 walkthrough | enforce | 13,083 | 652 (5.0%) | 12,431 | 29 | 0 | yes |
| ABCD | enforce | 2,008 | 1,390 (69.2%) | 618 | 68 | 0 | yes |
| ADE | shadow | 10,476 | 69 (0.7%) | 10,407 | 4 | 0 | yes |
| ToxicChat | enforce | 4,607 | 872 (18.9%) | 3,735 | 44 | 0 | yes |

With the edge tier in the rules function, a split pays where the edge settles much: it answers about 70%
of ABCD and a fifth of ToxicChat without the encoder. ADE's edge settles almost nothing here: the head
fails its calibration gate with every edge candidate (see [Release pipeline](#release-pipeline)), so
training falls back to the most cautious one. (The Banking77 row is the walkthrough's release, which has
no edge tier: only its rules answer.)

`npm run serve:bench` measures each function from a cold start, in a fresh process: module load, model
load, release load and validation, the first request, resident memory, and latency (the model function
embedding every request for real). Measured on 16 October on this machine, idle (it is not a function's
CPU). Every release uses EmbeddingGemma (8-bit). The three release simulations carry the edge tier, so
their rules function also loads the edge's static encoder; the walkthrough's Banking77 release has none.

| Release | Function | Import | Model load | Release load | First request | Memory after loading | Median | p95 |
|---|---|---|---|---|---|---|---|---|
| Banking77 (walkthrough, no edge tier) | rules | 43 ms | - | 6 ms | 2 ms | 71 MB | 4 µs | 7 µs |
| | model | 44 ms | 1.17 s | 17 ms | 0.12 s | 714 MB | 50 ms | 63 ms |
| ABCD (edge tier, kNN heads) | rules | 44 ms | - | 0.62 s | 5 ms | 127 MB | 67 µs | 141 µs |
| | model | 44 ms | 1.17 s | 0.67 s | 0.13 s | 824 MB | 54 ms | 73 ms |
| ADE (edge tier, linear head) | rules | 43 ms | - | 0.59 s | 5 ms | 133 MB | 85 µs | 168 µs |
| | model | 75 ms | 1.15 s | 0.66 s | 0.13 s | 802 MB | 62 ms | 87 ms |
| ToxicChat (edge tier, linear heads) | rules | 44 ms | - | 0.62 s | 6 ms | 174 MB | 50 µs | 508 µs |
| | model | 53 ms | 1.19 s | 0.74 s | 0.13 s | 821 MB | 57 ms | 0.16 s |

- **Rules function.** It needs 70-175 MB and is ready in under 0.7 s, so it still fits small functions and
  edge runtimes. The edge tier's static encoder accounts for most of that: about 0.6 s of release load, and
  50-85 µs per message (Banking77, without an edge, answers in 4 µs).
- **Model function.** Its cold start is the model load, about 1.2 s for the 309 MB 8-bit EmbeddingGemma,
  and it holds 0.7-0.8 GB after loading; size it at about 1.5 GB for these releases. Embedding one message
  takes 50-62 ms here, about five times mpnet's 8-bit 10 ms (measured on 8 October). That is the latency
  price of EmbeddingGemma's accuracy, and it is the main reason to let the rules function settle what it can.
- Real Lambda numbers depend on the configured memory (CPU share scales with it): measure there before
  sizing.

## Release pipeline

`src/release.ts` is a reference training pipeline, from a sample of real traffic to a release you can
deploy:

1. **`plan`** - embed the traffic, score it (with an independent offline teacher when one is
   available), draw a stratified probability sample and write one blinded labelling queue.
2. **`train`** - apply the reviewers' labels, mine and certify firing rules and joint dismissal rules,
   train the edge tier (`trainEdge`, on by default: the rules plus lexical and static-embedding scorers,
   certified to fire and dismiss on the calibration sample's design), train each head behind it (`type: 'auto'` picks linear, kNN or a stack per label; thresholds carry the
   strongest guarantee the data supports; optional caps on how often a head may fire), then build and
   validate the artifact.
3. **Release** - a complete router release in `release/`: `manifest.json`, `classifier.json`,
   `rules.json`, `edge.json`, `policy.json` and `evidence.json` (the final routing and per-head evaluation on the
   held-out test split, with each head's promise as its acceptance criterion), plus `deploy.json`,
   `summary.json` and `report.md`. `loadRelease` serves it with `loadRouter`: validated, paired, and
   refused if the encoder's identity differs (the identity's revision is the ONNX file's digest).

`src/release-simulate.ts` runs the pipeline on a benchmark with a simulated reviewer, serves the release
with the router (enforce mode when the release supports it, else shadow) and routes every message of the
benchmark's fully labelled future file. It checks each head's promise from the router's telemetry, the
final routing, and every router decision against the reference evaluator:

```bash
npm run release:simulate -- abcd           # or ade, toxicchat
npm run export:onnx                        # only for RELEASE_ENCODER=mpnet-layers: the multi-layer model (needs Python with torch, transformers, onnx, onnxruntime)
```

The pipeline embeds with EmbeddingGemma (`src/embed-gemma.ts`), the default.
`RELEASE_ENCODER=mpnet-layers` uses all-mpnet-base-v2 layers 4, 8 and 12 instead, mean-pooled and concatenated through the ONNX export
(`src/embed-layers.ts`). Either way it truncates exactly as the training library does, so training and
serving features match. Acceptance results (`out/release-sim/<dataset>/acceptance.md`, seed 7):

| Release | Head | Promise | Met on the future file | Recall | False alarms | Precision |
|---|---|---|---|---|---|---|
| ABCD support chats | account access (kNN head) | recall ≥ 0.85 | yes | 0.876 | 1.60% | 0.859 |
| | refund (kNN head) | precision ≥ 0.80 | yes (0.925) | 0.725 | 0.31% | 0.925 |
| ADE (adverse drug events) | adverse event | recall ≥ 0.85 | yes | 0.908 | 23.7% | 0.505 |
| ToxicChat | toxic | recall ≥ 0.80 | yes | 0.917 | 7.73% | 0.473 |
| | jailbreak | recall ≥ 0.70 | yes | 0.893 | 3.05% | 0.327 |

Every promise holds on the future file, and the router's decisions match the reference evaluator on
every message (0 mismatches in 2,008, 10,476 and 4,607 decisions). ABCD and ToxicChat load in enforce
mode. **ADE loads in shadow mode only:** its probabilities are overconfident where decisions are made.
In the top tenth of test scores they predict 89.8% positive and 68.2% are (±5.8 points), and the
calibration gate refuses the head (corrected Cox test p = 0.0012). The recall promise rests on ranks, not
on the probabilities, so it still holds on the future file. That is the gate working as designed: the
probabilities shown to reviewers would overstate how sure the head is. Promote it after recalibrating on
more labels. How many: for a label as diffuse as ADE's (about 20% positive, about 140 positives among 720
calibration rows), the lab redrew calibration samples around the seed-7 head 200 times per size. At 720
calibration rows, the release's default, about 1 draw in 20 (5.5%) left the top tenth's probabilities 5
points or more off the truth; at 1,440 rows, 1 in 100. `extraCalibrationLabels: 0.4` buys that: 720 more
labels, all to calibration, with train and test unchanged. In the lab's release simulations on five more seeds (8-12), the calibration gate never
refused ADE's head on EmbeddingGemma: this is the seed-7 draw, not the encoder. With every edge candidate
failing, training also falls back to the most cautious edge, which settles 0.7% of ADE.

The edge tier settles 69.2% of the ABCD future file without the main encoder (the release estimated
70.1%), 0.7% of ADE's (0.6%) and 18.9% of ToxicChat's (20.5%); before the edge tier, the rules alone
settled 48.2%, 0.7% and none. After priority and dismissal the router routes 14.2% of ABCD messages
(88% to an acceptable route), 37.7% of ADE's (51%) and 14.2% of ToxicChat's (38%). Routing costs about 0.1-0.15 ms per message
for linear heads and about 2 ms for ABCD's kNN heads, which compare each message with the stored training
reference (embedding time excluded). Results from 16 October, on EmbeddingGemma and the libraries in this
workspace.

**Reviewer errors.** The guarantees assume the reviewers' labels are correct. In the lab's simulation, 5%
of labels were flipped, at random or on the hardest cases, and 10% on the hardest cases. No promise failed silently: every
head that kept its guarantee also kept its promise. The errors cost in two other ways:
- **Refused guarantees.** A mislabelled positive scores like a negative, so reaching the recall target
  means firing on too much traffic, and the gates refuse the head. Banking77's `urgent` (rare and cleanly
  separated) lost every guarantee at 5% errors.
- **More false alarms** where a guarantee survives: about 2.5 to 3 points on ABCD `access` and ToxicChat
  `toxic`.

Raising the target to allow for the expected error rate (T / (1 - ε)) made both worse, so the pipeline
does not do it. If many heads are refused for no clear reason, check label quality first.

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

- **Benchmark** (EmbeddingGemma, the default encoder): average precision 0.862 for `toxic` (0.802 on mpnet), between the dataset's published
  ToxicChat-T5-large (0.885, fine-tuned) and OpenAI Moderation (0.631); 0.862 for `jailbreak` (0.858 on mpnet).
- **`toxic`:** stated recall ≥ 0.8 (design, linearised) and reached 0.917 (0.881–0.942) on the test
  file, with precision 0.50 and false alarms 6.9%. The guarantee is about traffic like the sampled
  population; the test file is a later period, and a shift between them can move recall below the target.
- **`jailbreak`:** with 20 calibration positives the head fails its gates: recall ≥ 0.7 needs a
  threshold below what its 8% firing cap allows. (The release pipeline's multi-layer heads, with a
  teacher-guided sample, do support a `jailbreak` promise: see [Release pipeline](#release-pipeline).)
- **Rules:** 4 certified, catching 8.6% of toxic prompts at 85% precision. Toxicity is less lexical
  than banking intents.
- **Monitor:** quiet on a random production window. The test file is ordered: its first 1,200 rows
  are 20% toxic, and that window raised an alert.

### Teacher labels

`teacher-labels/` holds an offline teacher's answers (Clef on Cloudflare Workers AI) for ToxicChat
and Banking77, used to choose and stratify what people label. See `manifest.json` there.
- **Training-only:** they never calibrate, set thresholds or certify.
- **No prompt text:** each entry is keyed by a hash of the model, the questions and the prompt.
- **Reruns cost nothing:** `npm run teacher:toxicchat -- [--model clef]` sends only prompts that are
  missing. It needs Cloudflare credentials in `~/.config/liquidau/cloudflare.env`.
- **Before training on a teacher's labels, check its calibration against your human labels.** In
  the lab (stage JV), TypeSafe's Jev (lab only; billed from AI Gateway credit) labelled ToxicChat
  well against ToxicChat's own labels (ECE 0.014, against Clef's 0.108). Heads trained only on its labels
  matched human-trained heads there. On Banking77 `refund`, where its yes labels were right only 36%
  of the time, heads trained on its labels lost 19 points of recall. A teacher that ranks well can
  still label badly for training.

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

## Showcase page

`npm run site` (after `npm start` and the release simulations) writes `site/index.html`, a single
self-contained page. Every number on it comes from the release files or from routing each release's
labelled check file through the router.

**The pathways diagram** comes first. It follows a message through the edge tier (fire, dismiss or pass
on) and the model tier (route, review band or no route). For the selected dataset it shows:
- the share of messages on each pathway, with line widths in proportion;
- how often each pathway's decisions were correct;
- real messages that took each pathway. ToxicChat prompts labelled toxic are withheld.

**A card per dataset** follows, showing:
- the encoder, the edge's scorers, the labels spent, and whether the release enforces or runs in shadow
  mode (with the gate that failed);
- for each head: its definition, model type, guarantee and promise, and the measured value at release;
- each head's own recall, precision and false alarms on the check file, with 95% Wilson intervals,
  and whether it kept its promise;
- final routing, after the policy's priority picks one route per message.

CFPB has no release, because the gates refuse it, so its card carries the lab's figures and says so.
The check files are held out from everything each release was built from. Only CFPB's is a later period
in time.

## Operations

`src/ops-monitor.ts` turns router telemetry (from one router, or both functions of a split deployment)
into windows and checks each live window against a reference window: the release's own traffic, routed
by the same release. Scores are compared per head in two strata - requests the router always scores,
and the monitoring sample of rules-settled requests - so no inverse weighting is needed; the share the
rules and the edge settle, each label's rule firing, the edge's fires and dismissals per label, and
unavailable results are compared as rates. All checks share
a 1% family-wise false-alarm rate per window. A drift alarm recommends retraining; an alarm on
unavailable results says to investigate the service instead.

`npm run ops:check` on the ABCD release (reference: its sampling frame):

| Window | Requests | Settled by the rules and the edge | Alerts | Retrain? | Investigate? |
|---|---|---|---|---|---|
| Future file, first half (a later period) | 1,000 | 70.0% | none | no | no |
| Future file, second half | 1,008 | 68.5% | none | no | no |
| Account-access requests surging to 35% | 1,000 | 71.4% | the edge fires `access` more and dismisses it less | yes | no |
| Encoder failing one call in five | 1,000 | 70.0% | unavailable results up (56: only the requests that reach the encoder can fail) | no | yes |

With the edge settling most requests, the surge shows up in the edge's own decisions rather than in the
encoder's scores. A score stratum with fewer than 30 scores in either window is skipped and listed; at
a 5% monitoring rate that needs about 600 edge-settled requests per window, so use longer windows, or a
higher monitoring rate, where edge-settled traffic matters.

**Prevalence.** `estimatePrevalence` estimates each head's prevalence in a window from its routing rate
alone (black-box shift estimation, BBSE). It corrects the routing rate with the release's own routing on
its labelled test split (`evidence.json`) and gives a 95% interval. On the ABCD windows (the truth is known
here only because the simulation's files are fully labelled):

| Window | `access`: routed / estimated (95%) / true | `refund`: routed / estimated (95%) / true |
|---|---|---|
| Future file, first half | 10.4% / 10.8% (8.5-13.0) / 10.2% | 4.5% / 5.8% (3.5-8.1) / 5.4% |
| Future file, second half | 10.0% / 10.3% (8.2-12.5) / 9.8% | 3.5% / 4.3% (2.3-6.3) / 4.8% |
| Account-access surge | 31.4% / 33.5% (29.8-37.3) / 35.0% | 2.9% / 3.5% (1.7-5.3) / 3.1% |
| Encoder failing one call in five | 10.6% / 11.0% (8.7-13.3) / 10.2% | 4.4% / 5.7% (3.4-8.1) / 5.4% |

- **Validation.** Every interval contains the truth. In the lab (7 heads, 3 kinds of window, 51 cases
  each), the intervals always covered, and the estimate was 0.7 to 1.0 points off on average, against
  2.9 to 4.5 for the raw firing rate.
- **Assumption.** BBSE assumes label shift: the mix of positives and negatives changes, but each class
  looks as it did at release. When the messages themselves change (a new product, a new kind of abuse),
  the estimate is not validated. The drift checks above are what catch that.
- **What it is for: precision promises.** A recall promise does not depend on prevalence. A precision
  promise does, because when positives become rarer, the same false alarms make up a larger share of
  what fires. When the interval sits well below the release's prevalence, raise each precision head's
  threshold to the lowest one whose precision, at the interval's lower end, meets the target. In the
  lab, at half prevalence, the shipped threshold missed ABCD `refund`'s precision promise in 5 of 9
  seeds; the raised threshold held in all 9. Halving prevalence cost a fifth to a quarter of the head's
  recall. The test rates are unweighted, which suits ABCD's equal-probability sample. A release whose
  sample over-represents some strata needs design-weighted rates.

**Runbook.**
- **Release.** `npm run release:simulate` (or your pipeline) writes a complete release; its
  `summary.json` lists why it may only run in shadow mode, if it may.
- **Shadow, then promote.** Deploy the new manifest digest in shadow mode alongside production and
  compare its candidates with production's decisions on live telemetry. Promote by redeploying the
  same digest in enforce mode; `loadRouter` and `loadRulesTier` refuse enforcement when the release's
  gates or evidence don't support it.
- **Split deployments.** Deploy the model function first, then the rules function, both on the new
  digest. In between, forwards are answered `RELEASE_MISMATCH` (retryable): retry once both serve it.
- **Roll back** by redeploying the previous digest (both functions). A long-running process can swap
  without mixing releases with `RouterSlot.reload`.
- **Monitor** every window with `checkWindow`. Drift (scores, rule firing, the rules' share) means
  the release no longer fits the traffic: label a fresh sample (`plan`), train, release, shadow,
  promote. Unavailable results mean the service is failing (encoder, release source): retraining
  won't help. Watch the router's `telemetry().dropped` too: dropped events weaken every check.
  Track each head's prevalence (`estimatePrevalence`) and re-threshold precision heads when it falls.

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
