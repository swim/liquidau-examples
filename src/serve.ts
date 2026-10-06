/**
 * A serving handler shaped for serverless functions: everything heavy is loaded once per instance
 * (module scope in a real function), and each request runs rules, then the embedding, the heads and
 * the decision. Imports only the package roots, which use no Node built-ins, so the same code runs
 * on edge runtimes when `embed` calls a hosted embedding API.
 *
 * Policy: rules OR classifier. A certified rule's hit counts as its head firing, and goes through
 * `decide`, so priority and suppression still apply. When the hit's head is first in priority and
 * no suppression rule can silence it, the decision can't change whatever the scores are, so the
 * embedding is skipped. For monitoring, a seeded share of those skipped messages is still embedded,
 * so score windows sample all traffic, not only what no rule caught.
 */
import {
  decide, missingPolicyHeads, scoreEmbedding, truncateText, validateArtifact,
  type ClassifierArtifact, type Decision, type DecisionPolicy, type Scores,
} from '@liquidau/embedding-classifier';
import { ruleSetMatcher, validateRuleSet } from '@liquidau/rule-miner';

export interface ServeOptions<H extends string> {
  artifact: unknown;
  ruleSet: unknown;
  policy: DecisionPolicy<H>;
  /** The embedding call: the same model and settings the artifact records. */
  embed: (text: string) => Promise<number[]>;
  /** The truncation used at training time (CachedEmbedder's maxChars), if any. */
  maxChars?: number;
  /** Share of rule-decided messages still embedded for monitoring (default 0.05). */
  monitorShare?: number;
  /** Deterministic sampling for monitorShare, e.g. a hash of the message id (default Math.random). */
  sample?: () => number;
}

export interface Served<H extends string> extends Decision<H> {
  /** The certified rule that fired, if any. */
  rule: string | null;
  /** False when a rule decided and the embedding was skipped. */
  embedded: boolean;
  /** The heads' scores, when embedded: log them for monitorWindow. */
  scores?: Scores<H>;
}

export function createHandler<H extends string>(options: ServeOptions<H>) {
  const { policy, embed, maxChars, monitorShare = 0.05, sample = Math.random } = options;
  const model = validateArtifact(options.artifact as ClassifierArtifact<H>) as ClassifierArtifact<H>;
  const missing = missingPolicyHeads(model, policy);
  if (missing.length) throw new Error(`policy heads missing from the artifact: ${missing.join(', ')}`);
  const matcher = ruleSetMatcher(validateRuleSet(options.ruleSet));
  const top = policy.priority[0];
  const unsuppressable = (head: string) => !(policy.suppress ?? []).some((s) => (s.heads as readonly string[]).includes(head));

  return async function handle(text: string): Promise<Served<H>> {
    const hit = matcher.match(text);
    const decisive = hit !== null && hit.label === top && unsuppressable(hit.label);
    if (decisive && sample() >= monitorShare) return { head: hit.label as H, reason: 'above_threshold', rule: hit.id, embedded: false };
    const scores = scoreEmbedding(model, await embed(truncateText(text, maxChars)));
    // Rules OR classifier: a rule hit lifts its head to its threshold before the policy runs.
    const forced: Scores<H> = { ...scores };
    const head = hit?.label as H | undefined;
    if (head !== undefined && model.heads[head]) forced[head] = Math.max(scores[head] ?? 0, model.heads[head]!.threshold);
    return { ...decide(model, forced, policy), rule: hit?.id ?? null, embedded: true, scores };
  };
}
