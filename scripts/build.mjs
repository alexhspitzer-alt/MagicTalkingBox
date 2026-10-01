import { build } from 'vite';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
await build();
async function filesIn(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  return (await Promise.all(entries.map(entry => entry.isDirectory() ? filesIn(path.join(directory, entry.name)) : path.join(directory, entry.name)))).flat();
}
const files = (await filesIn('dist')).sort();
const hash = createHash('sha256');
for (const file of files) hash.update(await readFile(file));
const cache = `magic-box-app-${hash.digest('hex').slice(0, 16)}`;
const urls = files.map(file => './' + path.relative('dist', file).split(path.sep).join('/'));
// Precache the entire bundled app, including the inference web worker. No CDN
// scripts/fonts are needed after installation. Model storage is owned by WebLLM.
await writeFile('dist/service-worker.js', `
const CACHE = ${JSON.stringify(cache)};
const FILES = ${JSON.stringify(urls)};
self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key.startsWith('magic-box-app-') && key !== CACHE).map(key => caches.delete(key)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== self.location.origin || !url.href.startsWith(self.registration.scope)) return;
  event.respondWith(caches.open(CACHE).then(async cache => {
    const cached = await cache.match(event.request);
    if (cached) return cached;
    if (event.request.mode === 'navigate') return await cache.match(new URL('index.html', self.registration.scope)) || fetch(event.request);
    return fetch(event.request);
  }));
});
`);
console.log(`Offline app cache: ${files.length} files, ${cache}`);
