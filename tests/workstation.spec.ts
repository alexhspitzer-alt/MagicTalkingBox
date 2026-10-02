import { test, expect } from '@playwright/test';

test('unsupported devices get a clear error and cannot download a model', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
  await page.goto('./');
  await expect(page.locator('#error')).toContainText('does not support WebGPU');
  await expect(page.locator('#load')).toBeDisabled();
  await expect(page.locator('#send')).toBeDisabled();
  await expect(page.locator('#size')).toContainText('194 MiB');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('app and saved chat survive an offline reload on a repository subpath', async ({ page, context }) => {
  await page.goto('./');
  await expect(page.locator('#app-cache')).toHaveText('App cache: complete');
  await page.evaluate(() => localStorage.setItem('magic-talking-box.chat.v1', JSON.stringify([{ role: 'user', content: '<b>local history</b>' }, { role: 'assistant', content: 'Saved reply' }])));
  // Model the browser connection signal as well as blocking network traffic.
  // Some headless builds keep navigator.onLine=true under CDP emulation.
  await page.addInitScript(() => Object.defineProperty(navigator, 'onLine', { get: () => false }));
  await context.setOffline(true);
  await page.reload();
  await expect(page.locator('h1')).toContainText('Magic Talking Box');
  await expect(page.locator('#messages')).toContainText('<b>local history</b>');
  await expect(page.locator('#messages b')).toHaveCount(0);
  await expect(page.locator('#messages')).toContainText('Saved reply');
  await expect(page.locator('#network')).toContainText('Offline');
  // The inference worker itself must also be available without HTTP caching.
  expect(await page.evaluate(async () => {
    const cache = await caches.open((await caches.keys()).find(key => key.startsWith('magic-box-app-'))!);
    const worker = (await cache.keys()).find(key => /\/assets\/worker-.*\.js$/.test(key.url));
    return Boolean(worker && (await fetch(worker.url)).ok);
  })).toBe(true);
});

test('offline readiness requires every weight shard and runtime artifact', async ({ page }) => {
  await page.goto('./');
  await expect(page.locator('#app-cache')).toHaveText('App cache: complete');
  await page.evaluate(async () => {
    const base = 'https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q4f32_1-MLC/resolve/main/';
    const model = await caches.open('webllm/model');
    await model.put(base + 'tensor-cache.json', Response.json({ records: [{ dataPath: 'test-shard.bin', nbytes: 42 }] }));
    await model.put(base + 'tokenizer.json', new Response('{}'));
    await (await caches.open('webllm/config')).put(base + 'mlc-chat-config.json', Response.json({ tokenizer_files: ['tokenizer.json'] }));
    await (await caches.open('webllm/wasm')).put('https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_84/base/SmolLM2-360M-Instruct-q4f32_1_cs1k-webgpu.wasm', new Response('test-runtime'));
  });
  await page.reload();
  await expect(page.locator('#model-cache')).toHaveText('Model cache: incomplete');
  await expect(page.locator('#offline-status')).not.toContainText('Offline ready');
  await page.evaluate(async () => {
    await (await caches.open('webllm/model')).put('https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q4f32_1-MLC/resolve/main/test-shard.bin', new Response('test-weights'));
  });
  await page.reload();
  await expect(page.locator('#model-cache')).toHaveText('Model cache: complete');
  await expect(page.locator('#offline-status')).toContainText('Offline ready');
});

test('real model chats offline and reloads with zero remote downloads', async ({ page, context }) => {
  test.skip(!process.env.REAL_MODEL, 'Opt-in: downloads ~200 MB and needs a working WebGPU adapter.');
  test.setTimeout(1_200_000);
  page.on('console', message => console.log(message.type(), message.text()));
  await page.goto('./');
  await expect(page.locator('#load')).toBeEnabled();
  await page.evaluate(() => {
    let last = '';
    const log = () => {
      const status = document.querySelector('#status')!.textContent || '';
      if (status !== last) { console.log(status); last = status; }
    };
    new MutationObserver(log).observe(document.querySelector('#status')!, { childList: true });
    new MutationObserver(() => console.error(document.querySelector('#error')!.textContent)).observe(document.querySelector('#error')!, { childList: true });
  });
  await page.locator('#load').click();
  await expect(page.locator('#send')).toBeEnabled({ timeout: 900_000 });
  await expect(page.locator('#offline-status')).toContainText('Offline ready');
  await context.setOffline(true);
  await page.locator('#prompt').fill('Reply with just the word hello.');
  await page.locator('#send').click();
  await expect(page.locator('#send')).toBeEnabled({ timeout: 300_000 });
  await expect(page.locator('.assistant .message-content')).not.toBeEmpty();
  console.log('First offline reply:', await page.locator('.assistant .message-content').innerText());
  await expect(page.locator('#error')).toBeHidden();
  await page.reload();
  await expect(page.locator('#load')).toBeEnabled();
  let remoteRequests = 0;
  page.on('request', request => { if (!request.url().startsWith('http://127.0.0.1:4173/')) remoteRequests++; });
  await page.locator('#load').click();
  await expect(page.locator('#send')).toBeEnabled({ timeout: 300_000 });
  await page.locator('#prompt').fill('What is two plus two? Answer briefly.');
  await page.locator('#send').click();
  await expect(page.locator('#send')).toBeEnabled({ timeout: 300_000 });
  await expect(page.locator('.assistant .message-content').last()).not.toBeEmpty();
  await expect(page.locator('#error')).toBeHidden();
  expect(remoteRequests).toBe(0);
});

test('real cache loads the model after an offline page reload', async ({ page, context }) => {
  test.skip(!process.env.REAL_CACHE, 'Opt-in: downloads the actual model, but does not require generation.');
  test.setTimeout(600_000);
  await page.goto('./');
  await expect(page.locator('#load')).toBeEnabled();
  await page.locator('#load').click();
  await expect(page.locator('#send')).toBeEnabled({ timeout: 300_000 });
  await expect(page.locator('#offline-status')).toContainText('Offline ready');
  await context.setOffline(true);
  await page.reload();
  await expect(page.locator('#load')).toBeEnabled();
  let remoteRequests = 0;
  page.on('request', request => { if (!request.url().startsWith('http://127.0.0.1:4173/')) remoteRequests++; });
  await page.locator('#load').click();
  await expect(page.locator('#send')).toBeEnabled({ timeout: 300_000 });
  await expect(page.locator('#error')).toBeHidden();
  expect(remoteRequests).toBe(0);
});

test('model menus switch modes without downloading weights', async ({ page }) => {
  const remote: string[] = [];
  page.on('request', request => { if (/huggingface|githubusercontent|jsdelivr/.test(request.url())) remote.push(request.url()); });
  await page.goto('./');
  await expect(page.locator('#model-select option')).toHaveCount(7);
  await page.locator('#model-select').selectOption('Qwen2.5-1.5B-Instruct-q4f32_1-MLC');
  await expect(page.locator('#model')).toContainText('Qwen2.5 1.5B');
  await expect(page.locator('#size')).toContainText('828 MiB');
  for (const [id, size, allocation] of [
    ['Qwen2.5-3B-Instruct-q4f32_1-MLC', '1,656 MiB', '2,894 MB'],
    ['Llama-3.2-3B-Instruct-q4f32_1-MLC', '1,724 MiB', '2,952 MB'],
    ['Qwen2.5-7B-Instruct-q4f32_1-MLC', '4,086 MiB', '5,900 MB'],
  ]) {
    await page.locator('#model-select').selectOption(id);
    await expect(page.locator('#size')).toContainText(size);
    await expect(page.locator('#model-description')).toContainText(allocation);
  }
  await page.locator('#model-select').selectOption('Qwen2.5-1.5B-Instruct-q4f32_1-MLC');
  await page.locator('#mode-select').selectOption('image');
  await expect(page.locator('#model-select option')).toHaveCount(3);
  await expect(page.locator('#send')).toHaveText('Draw ↗');
  await expect(page.locator('#context')).toContainText('512 × 512');
  await page.locator('#model-select').selectOption('janus-pro-1b');
  await expect(page.locator('#context')).toContainText('576 image tokens');
  await page.locator('#mode-select').selectOption('text');
  await expect(page.locator('#model-select')).toHaveValue('Qwen2.5-1.5B-Instruct-q4f32_1-MLC');
  await expect(page.locator('#send')).toBeDisabled();
  expect(remote).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('image UI saves PNGs, restores offline, and releases the worker on model switching', async ({ page, context }) => {
  // Exercise the worker protocol and storage with a tiny PNG; no AI inference.
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'gpu', { value: { requestAdapter: async () => ({ info: { description: 'Test GPU' }, requestDevice: async () => ({ destroy() {} }) }) } });
    let terminated = 0;
    const OriginalWorker = window.Worker;
    class ImageWorker extends EventTarget {
      constructor(url: string | URL, options?: WorkerOptions) {
        super();
        if (!String(url).includes('image-worker-')) return new OriginalWorker(url, options);
      }
      postMessage(message: { id: number; type: string }) {
        setTimeout(() => {
          if (message.type === 'load') this.dispatchEvent(new MessageEvent('message', { data: { id: message.id, type: 'result' } }));
          else {
            const bytes = Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aDAAAAAAASUVORK5CYII='), c => c.charCodeAt(0));
            this.dispatchEvent(new MessageEvent('message', { data: { id: message.id, type: 'result', result: { blob: new Blob([bytes], { type: 'image/png' }), seconds: 2 } } }));
          }
        }, 10);
      }
      terminate() { terminated++; (window as unknown as { terminatedWorkers: number }).terminatedWorkers = terminated; }
    }
    window.Worker = ImageWorker as unknown as typeof Worker;
  });
  await page.goto('./');
  await expect(page.locator('#app-cache')).toHaveText('App cache: complete');
  await page.locator('#mode-select').selectOption('image');
  await page.locator('#load').click();
  await expect(page.locator('#send')).toBeEnabled();
  await page.locator('#prompt').fill('<b>a tiny drawing</b>');
  await page.locator('#send').click();
  await expect(page.locator('#status')).toHaveText('Ready. Image saved on this device.');
  await expect(page.locator('.image-result img')).toHaveCount(1);
  await expect(page.locator('.image-result b')).toHaveCount(0);
  await expect(page.locator('.download')).toHaveAttribute('download', /\.png$/);
  await page.locator('#model-select').selectOption('janus-pro-1b');
  await expect(page.locator('#send')).toBeDisabled();
  expect(await page.evaluate(() => (window as unknown as { terminatedWorkers: number }).terminatedWorkers)).toBe(1);
  await context.setOffline(true); await page.reload();
  await expect(page.locator('#mode-select')).toHaveValue('image');
  await expect(page.locator('#model-select')).toHaveValue('janus-pro-1b');
  await expect(page.locator('.image-result img')).toHaveCount(1);
  await expect(page.locator('.image-result')).toContainText('<b>a tiny drawing</b>');
  expect(await page.evaluate(async () => {
    const name = (await caches.keys()).find(name => name.startsWith('magic-box-app-'))!;
    const cache = await caches.open(name);
    const keys = await cache.keys();
    return ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm'].every(file => keys.some(key => key.url.endsWith('/ort/' + file)));
  })).toBe(true);
  await page.locator('#clear').click();
  await expect(page.locator('.image-result')).toHaveCount(0);
});

test('selected image cache needs every artifact and the real image worker starts offline', async ({ page, context }) => {
  await page.goto('./');
  await expect(page.locator('#app-cache')).toHaveText('App cache: complete');
  await page.evaluate(async () => {
    const cache = await caches.open('magic-box-image-models-v1');
    const url = 'https://huggingface.co/test/image.onnx';
    await cache.put(new URL('/__magic-box__/image-manifest-sd-turbo.json', location.origin), Response.json({ urls: [url], bytes: 100 }));
  });
  await page.locator('#mode-select').selectOption('image');
  await expect(page.locator('#model-cache')).toHaveText('Model cache: incomplete');
  await page.evaluate(async () => {
    await (await caches.open('magic-box-image-models-v1')).put('https://huggingface.co/test/image.onnx', new Response('tiny fixture'));
  });
  await context.setOffline(true); await page.reload();
  await expect(page.locator('#model-cache')).toHaveText('Model cache: complete');
  await expect(page.locator('#offline-status')).toContainText('Offline ready');
  // A deliberately unknown model reports an error before downloads/GPU work.
  expect(await page.evaluate(async () => {
    const cache = await caches.open((await caches.keys()).find(key => key.startsWith('magic-box-app-'))!);
    const url = (await cache.keys()).find(key => /\/assets\/image-worker-.*\.js$/.test(key.url))!.url;
    return new Promise<string>((resolve, reject) => {
      const worker = new Worker(url, { type: 'module' });
      worker.onerror = () => { worker.terminate(); reject(Error('Image worker failed to start')); };
      worker.onmessage = event => { worker.terminate(); resolve(event.data.error); };
      worker.postMessage({ id: 1, type: 'load', model: 'unknown-test-model' });
    });
  })).toContain('Unknown model');
});
