import { test, expect } from '@playwright/test';
import { imageAssetCache } from '../src/image-cache';

test('cached model URLs work without network and are revoked after success or failure', async () => {
  const previousLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
  const previousCaches = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  let matches = 0;
  Object.defineProperty(globalThis, 'location', { configurable: true, value: { href: 'https://example.test/', origin: 'https://example.test' } });
  Object.defineProperty(globalThis, 'caches', { configurable: true, value: { open: async () => ({
    match: async () => { matches++; return new Response(new Uint8Array([1, 2, 3])); },
    add: async () => { throw new Error('Unexpected network request'); },
  }) } });
  try {
    const cache = await imageAssetCache();
    for (const fail of [false, true]) {
      let temporaryURL = '';
      const load = cache.withModelURL('https://example.test/model.onnx', async url => {
        temporaryURL = url;
        expect(Array.from(new Uint8Array(await (await fetch(url)).arrayBuffer()))).toEqual([1, 2, 3]);
        if (fail) throw new Error('Session initialization failed');
        return 'session';
      });
      if (fail) await expect(load).rejects.toThrow('Session initialization failed');
      else expect(await load).toBe('session');
      await expect(fetch(temporaryURL)).rejects.toThrow();
    }
    expect(matches).toBe(2);
  } finally {
    for (const [name, descriptor] of [['location', previousLocation], ['caches', previousCaches]] as const) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  }
});
