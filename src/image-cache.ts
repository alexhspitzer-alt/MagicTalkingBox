// Shared by worker and UI. A completed manifest lists every artifact used by
// the pinned runtime, including tokenizers and configuration.
export const IMAGE_CACHE = 'magic-box-image-models-v1';
function manifestURL(id: string) { return new URL(`/__magic-box__/image-manifest-${id}.json`, location.origin).href; }
export async function imageCacheInfo(id: string): Promise<{ complete: boolean; bytes: number }> {
  const cache = await caches.open(IMAGE_CACHE);
  const response = await cache.match(manifestURL(id));
  if (!response) return { complete: false, bytes: 0 };
  const manifest = await response.json() as { urls: string[]; bytes: number };
  const keys = new Set((await cache.keys()).map(key => key.url));
  return { complete: manifest.urls.length > 0 && manifest.urls.every(url => keys.has(url)), bytes: manifest.bytes };
}
export async function imageAssetCache() {
  const cache = await caches.open(IMAGE_CACHE);
  const used = new Set<string>();
  const failures: string[] = [];
  const urlOf = (request: RequestInfo | URL) => request instanceof Request ? request.url : new URL(String(request), location.href).href;
  const tracked = {
    async match(request: RequestInfo | URL) {
      const url = urlOf(request);
      const response = await cache.match(url);
      if (response) used.add(url);
      return response;
    },
    async put(request: RequestInfo | URL, response: Response) {
      const url = urlOf(request);
      try { await cache.put(url, response); used.add(url); }
      catch (error) { failures.push(`${url}: ${String(error)}`); throw error; }
    },
  };
  return {
    tracked,
    async download(url: string) {
      let response = await tracked.match(url);
      if (!response) {
        // Stream directly to storage instead of cloning multi-GB responses.
        await cache.add(url);
        response = await tracked.match(url);
      }
      if (!response) throw new Error(`Model file could not be cached: ${url}`);
      return response.arrayBuffer();
    },
    async complete(id: string) {
      if (failures.length) throw new Error(`Model files could not be saved. Free browser storage and retry. ${failures[0]}`);
      const urls = [...used];
      let bytes = 0;
      for (const url of urls) {
        const response = await cache.match(url);
        if (!response) throw new Error('Browser storage evicted a model file during loading. Free storage and retry.');
        bytes += Number(response.headers.get('content-length')) || 0;
      }
      await cache.put(manifestURL(id), Response.json({ urls, bytes }));
    },
  };
}
