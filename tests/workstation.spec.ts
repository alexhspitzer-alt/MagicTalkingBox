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
