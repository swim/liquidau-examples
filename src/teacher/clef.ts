/**
 * Clef as an OFFLINE teacher: it labels training data in a batch; nothing at runtime calls it. Its
 * answers are training-only (provenance rule P4 in embedding-classifier's docs/DATA.md):
 * calibration, thresholds and certification still use human-labelled, probability-sampled records.
 *
 * Workers AI REST: POST https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}
 * with { model, state, questions } - up to 64 typed questions per request; `noul` questions return the
 * probability of yes. https://developers.cloudflare.com/workers-ai/models/clef-flash/
 *
 * Credentials: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN from the environment, or from
 * ~/.config/liquidau/cloudflare.env (KEY=VALUE lines, chmod 600). Never logged, never written to the
 * repository. Answers are cached on disk by (model, questions, text), so a rerun pays nothing, and a
 * token budget stops a run before it spends more than intended.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

/**
 * 'jev' is TypeSafe's Jev (a third-party model on Workers AI, `typesafe/jev`): the same state-and-questions
 * schema, called through the generic /ai/run endpoint and billed from the AI Gateway balance (or BYOK).
 */
export type ClefModel = 'clef' | 'clef-flash' | 'jev';
export interface NoulQuestion { type: 'noul'; instructions: string }
export type Questions = Record<string, NoulQuestion>;

export function credentials(): { account: string; token: string } {
  let account = process.env.CLOUDFLARE_ACCOUNT_ID, token = process.env.CLOUDFLARE_API_TOKEN;
  const file = join(homedir(), '.config', 'liquidau', 'cloudflare.env');
  if ((!account || !token) && existsSync(file)) {
    for (const line of readFileSync(file, 'utf8').split('\n')) {
      const m = line.match(/^\s*(CLOUDFLARE_ACCOUNT_ID|CLOUDFLARE_API_TOKEN)\s*=\s*"?([^"\s]+)"?\s*$/);
      if (m?.[1] === 'CLOUDFLARE_ACCOUNT_ID') account ??= m[2];
      if (m?.[1] === 'CLOUDFLARE_API_TOKEN') token ??= m[2];
    }
  }
  if (!account || !token) throw new Error(`Clef needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (environment, or ${file})`);
  return { account, token };
}

export interface ClefTeacherOptions {
  model: ClefModel;
  questions: Questions;
  /** Cache file (JSON). One per dataset keeps files small. */
  cachePath: string;
  /** Stop before spending more input tokens than this in one run (default 3 million). */
  maxTokens?: number;
  /** Texts are truncated to this many characters before being sent (default 8,000). */
  maxChars?: number;
  concurrency?: number;
  log?: (line: string) => void;
}

export class ClefTeacher {
  private cache: Record<string, Record<string, number>>;
  private spent = 0;
  constructor(private readonly o: ClefTeacherOptions) {
    this.cache = existsSync(o.cachePath) ? JSON.parse(readFileSync(o.cachePath, 'utf8')) : {};
  }

  private key(text: string): string {
    return createHash('sha256').update(JSON.stringify([this.o.model, this.o.questions, this.truncate(text)])).digest('hex');
  }
  private truncate(text: string) { return text.slice(0, this.o.maxChars ?? 8000); }

  private save() {
    mkdirSync(dirname(this.o.cachePath), { recursive: true });
    const tmp = `${this.o.cachePath}.tmp`;
    writeFileSync(tmp, JSON.stringify(this.cache));
    renameSync(tmp, this.o.cachePath);
  }

  /** The cached answer for a text, without calling the API (undefined if never labelled). */
  cached(text: string): Record<string, number> | undefined { return this.cache[this.key(text)]; }

  /** Probability of yes per question for each text (cached; only uncached texts are sent). */
  async label(texts: string[]): Promise<Array<Record<string, number>>> {
    const { account, token } = credentials();
    const jev = this.o.model === 'jev';
    const url = jev ? `https://api.cloudflare.com/client/v4/accounts/${account}/ai/run` : `https://api.cloudflare.com/client/v4/accounts/${account}/ai/run/@cf/cloudflare/${this.o.model}`;
    const missing = [...new Set(texts.filter((t) => !(this.key(t) in this.cache)))];
    const log = this.o.log ?? (() => {});
    log(`Clef ${this.o.model}: ${missing.length} new of ${texts.length} texts (${texts.length - missing.length} cached)`);
    let next = 0, done = 0;
    const budget = this.o.maxTokens ?? 3_000_000;
    const worker = async () => {
      while (next < missing.length) {
        if (this.spent >= budget) throw new Error(`token budget of ${budget} reached after ${done} texts; rerun to continue (cached answers are kept)`);
        const text = missing[next++];
        const body = JSON.stringify(jev ? { model: 'typesafe/jev', input: { state: this.truncate(text), questions: this.o.questions } } : { model: this.o.model, state: this.truncate(text), questions: this.o.questions });
        for (let attempt = 0; ; attempt++) {
          const res = await fetch(url, { method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body });
          if (res.ok) {
            type Output = { answers?: Record<string, unknown>; usage?: { input_tokens?: number } };
            const raw = await res.json() as { result?: Output & { state?: string; result?: Output } };
            // Jev (third-party, generic /ai/run) nests its output one level down, under a completion state.
            if (jev && raw.result?.state !== undefined && raw.result.state !== 'Completed') throw new Error(`jev: request state ${raw.result.state}`);
            const json = { result: jev && raw.result?.result ? raw.result.result : raw.result };
            const answers = json.result?.answers ?? {};
            const out: Record<string, number> = {};
            for (const q of Object.keys(this.o.questions)) {
              const a = answers[q];
              // Observed form: { type: 'noul', noul: 0.1093 }. Plain numbers and one-element arrays are accepted too.
              const p = typeof a === 'number' ? a
                : Array.isArray(a) && typeof a[0] === 'number' ? a[0]
                  : a && typeof a === 'object' && typeof (a as { noul?: unknown }).noul === 'number' ? (a as { noul: number }).noul : NaN;
              if (!Number.isFinite(p)) throw new Error(`unexpected answer for ${q}: ${JSON.stringify(a).slice(0, 200)}`);
              out[q] = p;
            }
            this.cache[this.key(text)] = out;
            this.spent += json.result?.usage?.input_tokens ?? 0;
            break;
          }
          if ((res.status === 429 || res.status >= 500) && attempt < (jev ? 12 : 6)) { await new Promise((r) => setTimeout(r, Math.min(60_000, 1000 * 2 ** attempt))); continue; }
          throw new Error(`${this.o.model} HTTP ${res.status}: ${(await res.text()).slice(0, 300)}${res.status === 402 && jev ? ' (Jev is billed from the AI Gateway balance: add credit, or use BYOK)' : ''}`);
        }
        if (++done % 200 === 0) { this.save(); log(`  labelled ${done}/${missing.length} (${this.spent.toLocaleString()} input tokens this run)`); }
      }
    };
    try { await Promise.all(Array.from({ length: this.o.concurrency ?? 4 }, worker)); } finally { this.save(); }
    log(`Clef ${this.o.model}: done; ${this.spent.toLocaleString()} input tokens this run`);
    return texts.map((t) => this.cache[this.key(t)]);
  }
}
