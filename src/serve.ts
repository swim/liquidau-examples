/**
 * A serving handler shaped for serverless functions, on @liquidau/router. The release is loaded and
 * validated once per instance (module scope in a real function): every document's digest, the rule
 * set's pairing with the classifier, the encoder's exact identity and, in enforce mode, the release's
 * gates and evaluation evidence. Each request is then routed rules-first; the embedding runs only when
 * a score could change the outcome.
 *
 * Imports only the router's package root, which uses no Node built-ins: the host supplies the release
 * source (a directory, an object store, bundled bytes) and the encoder (a local model or a hosted API).
 */
import { loadRouter, type Encoder, type ReleaseSource, type ReviewOptions, type RouteResult, type RouterEvent, type RouterMode } from '@liquidau/router';
import { staticEncoder } from './static-host.ts';

export interface ServeOptions {
  source: ReleaseSource;
  manifestKey: string;
  /** From deployment configuration: the release this instance serves. */
  manifestSha256: string;
  mode: RouterMode;
  /** Implements the release's exact encoder identity (loading refuses anything else). */
  encoder: Encoder;
  /** Share of rules-settled requests still embedded for monitoring (default 0.05). */
  monitoringRate?: number;
  observer?: (event: RouterEvent) => void;
  /** A review adapter: every review result is sent to it (advisory; the decision is unchanged). */
  review?: ReviewOptions;
}

export interface Handler {
  handle(text: string, requestId: string): Promise<RouteResult>;
  close(): Promise<void>;
}

export async function createHandler(options: ServeOptions): Promise<Handler> {
  const router = await loadRouter({
    source: options.source, manifestKey: options.manifestKey, expectedManifestSha256: options.manifestSha256, mode: options.mode, encoder: options.encoder,
    timeoutMs: 10_000, monitoring: { sampleRate: options.monitoringRate ?? 0.05, samplingSalt: 'per-deployment-secret' }, staticEncoder,
    ...(options.observer ? { observer: options.observer } : {}),
    ...(options.review ? { review: options.review } : {}),
  });
  return {
    async handle(text, requestId) {
      const result = await router.route({ text, requestId });
      // Monitoring samples and telemetry run after the result; a function must finish them before it returns.
      await router.flush();
      return result;
    },
    close: () => router.close(),
  };
}
