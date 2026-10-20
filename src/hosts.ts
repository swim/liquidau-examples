/**
 * Function handlers for a split deployment of a router release: a small rules function (Lambda or an
 * edge Worker) in front of a model function (Lambda, sized for the encoder's model). Both load the same
 * release, pinned by its manifest digest, once per instance. The core handlers are platform-neutral;
 * the Lambda and Worker wrappers below only translate requests and responses.
 *
 * The rules function answers what the rules settle and forwards the rest as JSON; a request it answers
 * and selects for monitoring is also sent to the model function, after the answer, for telemetry only.
 */
import { loadRouter, loadRulesTier, type Encoder, type ForwardedRequest, type ReleaseSource, type RouteResult, type RouterEvent, type RouterMode } from '@liquidau/router';
import { staticEncoder } from './static-host.ts';

/** One release, as a deployment pins it. Both functions must use the same values. */
export interface ReleasePin {
  source: ReleaseSource;
  manifestKey: string;
  /** From deployment configuration (e.g. an environment variable), never from next to the release. */
  manifestSha256: string;
  mode: RouterMode;
  monitoring: { sampleRate: number; samplingSalt: string };
  observer?: (event: RouterEvent) => void | Promise<void>;
}

/** Sends a forward to the model function and returns its answer (null for a 'monitor' forward). */
export type Sender = (forward: ForwardedRequest) => Promise<RouteResult | null>;

/** A handler's answer, and the work to finish after answering (monitoring, telemetry). */
export interface Handled {
  status: number;
  body: unknown;
  after: Promise<void>;
}

const bad = (message: string): Handled => ({ status: 400, body: { error: message }, after: Promise.resolve() });

export async function createRulesFunction(pin: ReleasePin, send: Sender) {
  const tier = await loadRulesTier({
    source: pin.source, manifestKey: pin.manifestKey, expectedManifestSha256: pin.manifestSha256, mode: pin.mode, timeoutMs: 10_000, monitoring: pin.monitoring,
    staticEncoder, ...(pin.observer ? { observer: pin.observer } : {}),
  });
  return {
    tier,
    async handle(body: unknown): Promise<Handled> {
      const b = body as { text?: unknown; requestId?: unknown } | null;
      if (!b || typeof b.text !== 'string' || typeof b.requestId !== 'string') return bad('expected { text, requestId }');
      const outcome = await tier.handle({ text: b.text, requestId: b.requestId });
      if (!outcome.answered) {
        // A model tier that can't be reached (or answers nothing) is a retryable unavailable result,
        // not a platform error: callers handle it like any other unavailable route.
        const result = await send(outcome.forward).catch(() => null);
        if (!result) {
          const unavailable: RouteResult = { releaseId: tier.releaseId, requestId: outcome.forward.requestId, embedded: false, mode: tier.mode, actionable: false, outcome: 'unavailable', code: 'ENCODER_FAILURE', retryable: true };
          return { status: 503, body: unavailable, after: tier.flush() };
        }
        return { status: 200, body: result, after: tier.flush() };
      }
      // The monitoring forward never changes the answer, so it is sent after it.
      const monitor = outcome.monitor;
      const after = (async () => { if (monitor) await send(monitor).catch(() => null); await tier.flush(); })();
      return { status: 200, body: outcome.result, after };
    },
  };
}

export async function createModelFunction(pin: ReleasePin, encoder: Encoder) {
  const router = await loadRouter({
    source: pin.source, manifestKey: pin.manifestKey, expectedManifestSha256: pin.manifestSha256, mode: pin.mode, encoder, timeoutMs: 10_000,
    monitoring: pin.monitoring, staticEncoder, ...(pin.observer ? { observer: pin.observer } : {}),
  });
  return {
    router,
    async handle(body: unknown): Promise<Handled> {
      const result = await router.routeForwarded(body as ForwardedRequest);
      return { status: 200, body: result, after: router.flush() };
    },
  };
}

/** A Sender over HTTP (e.g. the model function's Lambda function URL). */
export function httpSender(url: string, headers: Record<string, string> = {}): Sender {
  return async (forward) => {
    const res = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(forward) });
    if (!res.ok) throw new Error(`model function answered ${res.status}`);
    return (await res.json()) as RouteResult | null;
  };
}

type Handler = { handle(body: unknown): Promise<Handled> };

const parseBody = (raw: string | null | undefined, base64: boolean): unknown => {
  const text = base64 ? new TextDecoder().decode(Uint8Array.from(atob(raw ?? ''), (c) => c.charCodeAt(0))) : raw ?? '';
  return JSON.parse(text || 'null');
};

/**
 * AWS Lambda (function URL or API Gateway proxy event). `init` runs once per instance (cold start);
 * the handler awaits the after-work before returning, because a returned Lambda may be frozen.
 */
export function lambdaHandler(init: () => Promise<Handler>) {
  let ready: Promise<Handler> | null = null;
  return async (event: { body?: string | null; isBase64Encoded?: boolean }) => {
    ready ??= init();
    const fn = await ready;
    let body: unknown;
    try { body = parseBody(event.body, event.isBase64Encoded === true); } catch { return { statusCode: 400, body: JSON.stringify({ error: 'invalid JSON' }) }; }
    const out = await fn.handle(body);
    await out.after;
    return { statusCode: out.status, headers: { 'content-type': 'application/json' }, body: JSON.stringify(out.body) };
  };
}

/**
 * A Cloudflare Worker (or any fetch-style runtime): the answer is returned at once and the after-work
 * handed to the platform with ctx.waitUntil.
 */
export function workerHandler<Env>(init: (env: Env) => Promise<Handler>) {
  let ready: Promise<Handler> | null = null;
  return {
    async fetch(request: Request, env: Env, ctx: { waitUntil(p: Promise<unknown>): void }): Promise<Response> {
      ready ??= init(env);
      const fn = await ready;
      let body: unknown;
      try { body = await request.json(); } catch { return Response.json({ error: 'invalid JSON' }, { status: 400 }); }
      const out = await fn.handle(body);
      ctx.waitUntil(out.after);
      return Response.json(out.body, { status: out.status });
    },
  };
}
