/**
 * A release source over a directory, for @liquidau/router in Node (a function's bundled release, or a
 * local release). Imports only node:fs and the router, so a rules function stays small.
 */
import { readFile } from 'node:fs/promises';
import { resolve, sep } from 'node:path';

import { isSafeKey, type ReleaseSource } from '@liquidau/router';

/** Safe relative keys only, and never a path outside the directory. */
export function fileSource(dir: string): ReleaseSource {
  const base = resolve(dir);
  return {
    async read(key, { signal }) {
      if (!isSafeKey(key)) throw new Error(`unsafe key ${JSON.stringify(key)}`);
      const path = resolve(base, key);
      if (!path.startsWith(base + sep)) throw new Error(`${key} escapes the release directory`);
      return new Uint8Array(await readFile(path, { signal }));
    },
  };
}
