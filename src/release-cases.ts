/**
 * The example's releases and the labelled messages each is checked on: the walkthrough's Banking77
 * release (`npm start`; its held-out messages) and the release simulations' (`npm run release:simulate
 * -- <dataset>`; each dataset's future file: held out from everything the release was built from, though
 * only CFPB's, which has no release, is a real later period). Encoders read the embedding caches, so routing a whole
 * file measures the routing, not the model. Used by split-check and the site.
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import type { Encoder } from '@liquidau/router';

import { loadAbcd, loadAde } from './benchmarks.ts';
import { loadBanking77 } from './data.ts';
import type { ClassifierArtifact } from '@liquidau/embedding-classifier';

import { makeEmbedder, walkthroughEncoder } from './embed.ts';
import { releaseEmbedderFor } from './release-encoder.ts';
import { loadToxicChat } from './toxicchat.ts';

export const ROOT = join(import.meta.dirname, '..');

export interface LabelledMessage { id: string; text: string; truth: Readonly<Record<string, 0 | 1 | null | undefined>> }

export interface ReleaseCase {
  name: string;
  /** 'banking77' | 'abcd' | 'ade' | 'toxicchat'. */
  key: string;
  dir: string;
  /** Every message split-check routes. */
  texts: string[];
  /** The labelled messages the release is checked on, and what they are. */
  checked: { items: LabelledMessage[]; description: string };
  encoder: () => Promise<Encoder>;
}

export async function releaseCases(): Promise<ReleaseCase[]> {
  const cases: ReleaseCase[] = [];
  if (existsSync(join(ROOT, 'out', 'release', 'deploy.json'))) {
    const messages = await loadBanking77();
    const texts = messages.map((m) => m.text);
    const byId = new Map(messages.map((m) => [m.id, m]));
    const heldOut = existsSync(join(ROOT, 'out', 'heldout.json')) ? JSON.parse(readFileSync(join(ROOT, 'out', 'heldout.json'), 'utf8')) as Array<{ id: string; labels: Record<string, 0 | 1 | null> }> : [];
    cases.push({
      name: 'Banking77 walkthrough', key: 'banking77', dir: join(ROOT, 'out', 'release'), texts,
      checked: { items: heldOut.map((h) => ({ id: h.id, text: byId.get(h.id)!.text, truth: h.labels })), description: 'the walkthrough\'s held-out labelled messages (not used to train or calibrate; not a later period)' },
      encoder: async () => {
        const embedder = await makeEmbedder(() => {});
        await embedder.embedMany(texts);
        return walkthroughEncoder((t) => embedder.embedMany([...t]));
      },
    });
  }
  const layerCase = (name: string, key: string, items: LabelledMessage[], description: string): ReleaseCase => ({
    name, key, dir: join(ROOT, 'out', 'release-sim', key, 'release'), texts: items.map((i) => i.text), checked: { items, description },
    encoder: async () => {
      const artifact = JSON.parse(readFileSync(join(ROOT, 'out', 'release-sim', key, 'release', 'classifier.json'), 'utf8')) as ClassifierArtifact;
      const cache = await releaseEmbedderFor(artifact.embedding);
      await cache.embedMany(items.map((i) => i.text));
      return cache.encoder();
    },
  });
  const simulated = (key: string) => existsSync(join(ROOT, 'out', 'release-sim', key, 'release', 'deploy.json'));
  if (simulated('abcd')) cases.push(layerCase('ABCD', 'abcd', loadAbcd().future, 'ABCD\'s dev and test conversations, never used to build the release (no timestamps, so unseen rather than later)'));
  if (simulated('ade')) cases.push(layerCase('ADE', 'ade', loadAde().future, 'half of the corpus, held out by a seeded hash of identical-sentence groups (held out, not a later period)'));
  if (simulated('toxicchat')) {
    const prompts = await loadToxicChat();
    const seen = new Set(prompts.filter((p) => p.split === 'train').map((p) => p.group));
    cases.push(layerCase('ToxicChat', 'toxicchat', prompts.filter((p) => p.split === 'test' && !seen.has(p.group)), 'ToxicChat\'s official test file, played as a fully labelled production period, minus duplicates of training prompts (held out, not timestamped later)'));
  }
  return cases;
}
