/**
 * A complete @liquidau/router release from a trained artifact, its rule set and a policy: the final
 * routing and per-head evaluation on held-out labelled items, evidence that records them with the
 * acceptance criteria, and buildRelease (which validates everything the loader will). Platform-neutral:
 * it returns bytes; writing or uploading them is the host's job.
 */
import { scoreEmbedding, type ClassifierArtifact, type EdgeDocument } from '@liquidau/embedding-classifier';
import { buildRelease, createDecisionEvaluator, createEdgeTier, documentDigest, encodeJson, type BuiltRelease, type EncoderIdentity, type ReleaseEvidence, type RouterPolicy } from '@liquidau/router';
import type { StaticEmbedder } from '@liquidau/text-preprocessing';
import { buildEvidence, evaluateFinal, evaluatePerHead, type AcceptanceCriteria, type FinalItem, type FinalReport } from '@liquidau/router/evaluation';
import { ruleSetHash, type RuleSet } from '@liquidau/rule-miner';

export interface HeldOutItem {
  id: string;
  text: string;
  /** The embedding the artifact scores (the same features it was trained on). */
  x: readonly number[];
  labels: Readonly<Record<string, 0 | 1 | null | undefined>>;
}

export interface RouterReleaseInput {
  releaseId: string;
  createdAt: string;
  artifact: ClassifierArtifact;
  ruleSet: RuleSet;
  policy: RouterPolicy;
  /** Held out from training and calibration. */
  items: readonly HeldOutItem[];
  dataset: { id: string; split: string; description?: string };
  criteria: AcceptanceCriteria;
  embedding: EncoderIdentity;
  packages: Record<string, string>;
  maxInputUtf8Bytes?: number;
  /** The edge tier (trainEdge's document) and, if it uses one, the static encoder it was trained with. */
  edge?: { document: EdgeDocument; staticEncoder?: StaticEmbedder };
}

export interface RouterRelease {
  built: BuiltRelease;
  evidence: ReleaseEvidence;
  final: FinalReport;
}

/** One route per head, in the given priority, with optional suppression. */
export const policyFor = (heads: readonly string[], priority: readonly string[], suppress: ReadonlyArray<{ when: readonly string[]; heads: readonly string[] }> = []): RouterPolicy => ({
  schema: 'liquidau-router-policy/1',
  routes: heads.map((h) => ({ id: h, destination: `queue:${h}` })),
  priority: [...priority],
  suppress: suppress.map((s) => ({ when: [...s.when], heads: [...s.heads] })),
  onAbstain: 'none',
});

export async function buildRouterRelease(input: RouterReleaseInput): Promise<RouterRelease> {
  const { artifact, ruleSet, policy } = input;
  const heads = Object.keys(artifact.heads);
  // The rules and the edge, exactly as the router's edge tier evaluates them.
  const tier = createEdgeTier(ruleSet, input.edge?.document ?? null, policy.priority, input.edge?.staticEncoder);
  const model = { heads: Object.fromEntries(heads.map((h) => [h, { threshold: artifact.heads[h]!.threshold, reviewFloor: artifact.heads[h]!.review_floor }])) };
  const scored = input.items.map((it) => ({ ...it, rules: tier.evaluate(it.text).findings, scores: scoreEmbedding(artifact, it.x) as Record<string, number> }));
  // Final routing needs every head's label; per-head reports take whatever each item has.
  const final = evaluateFinal(createDecisionEvaluator(model, policy), scored
    .filter((it) => heads.every((h) => it.labels[h] === 0 || it.labels[h] === 1))
    .map((it): FinalItem => ({ id: it.id, rules: it.rules, scores: it.scores, acceptable: heads.filter((h) => it.labels[h] === 1) })));
  const perHead = evaluatePerHead(model, scored.map((it) => ({ rules: it.rules, scores: it.scores, labels: it.labels })));
  const documents = { classifier: encodeJson(artifact), rules: encodeJson(ruleSet), policy: encodeJson(policy), ...(input.edge ? { edge: encodeJson(input.edge.document) } : {}) };
  const evidence = buildEvidence({
    inputs: {
      classifierSha256: await documentDigest(documents.classifier), rulesSha256: await documentDigest(documents.rules), policySha256: await documentDigest(documents.policy), rulesSemanticHash: ruleSetHash(ruleSet),
      ...(documents.edge ? { edgeSha256: await documentDigest(documents.edge) } : {}),
    },
    dataset: input.dataset, final, perHead, criteria: input.criteria,
  });
  const built = await buildRelease({
    releaseId: input.releaseId, createdAt: input.createdAt, documents: { ...documents, evidence: encodeJson(evidence) },
    embedding: { ...input.embedding, layers: input.embedding.layers ? [...input.embedding.layers] : null }, serving: { maxInputUtf8Bytes: input.maxInputUtf8Bytes ?? 1 << 16 }, packages: input.packages,
  });
  return { built, evidence, final };
}
