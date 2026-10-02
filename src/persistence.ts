import { MODEL_ID } from './model';
import { textModel, modelBase } from './text-model';
export type Message = { role: 'user' | 'assistant'; content: string };
const CHAT_KEY = 'magic-talking-box.chat.v1';

export function readChat(): Message[] {
  const saved: unknown = JSON.parse(localStorage.getItem(CHAT_KEY) || '[]');
  if (!Array.isArray(saved) || !saved.every(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')) {
    throw new Error('Saved conversation is invalid. Use New chat to reset it.');
  }
  return saved;
}
export function saveChat(messages: Message[]): void { localStorage.setItem(CHAT_KEY, JSON.stringify(messages)); }
export function clearChat(): void { localStorage.removeItem(CHAT_KEY); }

export async function storageInfo(requestPersistence = false) {
  if (!navigator.storage) return { persisted: false, usage: undefined, quota: undefined };
  const persisted = requestPersistence && navigator.storage.persist
    ? await navigator.storage.persist() : await navigator.storage.persisted?.() ?? false;
  const estimate = await navigator.storage.estimate?.() ?? {};
  return { persisted, ...estimate };
}

// Verify EVERY shard plus the runtime, tokenizer and configuration before
// promising offline reloads (hasModelInCache covers only the weight cache).
// Cache names and URLs are checked against pinned WebLLM 0.2.85.
export async function modelCacheInfo(id = MODEL_ID): Promise<{ complete: boolean; bytes: number }> {
  const MODEL = textModel(id), MODEL_BASE = modelBase(id);
  if (!('caches' in globalThis)) return { complete: false, bytes: 0 };
  const weights = await caches.open('webllm/model');
  const manifest = await weights.match(new URL('tensor-cache.json', MODEL_BASE));
  if (!manifest) return { complete: false, bytes: 0 };
  const data = await manifest.json() as { records: { dataPath: string; nbytes: number }[] };
  if (!Array.isArray(data.records) || !data.records.length) return { complete: false, bytes: 0 };
  const keys = new Set((await weights.keys()).map(key => key.url));
  const bytes = data.records.reduce((sum, record) => sum + record.nbytes, 0);
  const shards = data.records.every(record => keys.has(new URL(record.dataPath, MODEL_BASE).href));
  const configCache = await caches.open('webllm/config');
  const configResponse = await configCache.match(new URL('mlc-chat-config.json', MODEL_BASE));
  const config = configResponse ? await configResponse.json() as { tokenizer_files: string[] } : null;
  const tokenizerFile = config?.tokenizer_files?.includes('tokenizer.json') ? 'tokenizer.json'
    : config?.tokenizer_files?.includes('tokenizer.model') ? 'tokenizer.model' : undefined;
  const tokenizer = tokenizerFile && keys.has(new URL(tokenizerFile, MODEL_BASE).href);
  const wasm = await (await caches.open('webllm/wasm')).match(MODEL.model_lib);
  return { complete: Boolean(shards && tokenizer && wasm), bytes };
}

export async function cacheApp(): Promise<void> {
  // The development server is intentionally not cached.
  if (import.meta.env.DEV) return;
  if (!('serviceWorker' in navigator)) throw new Error('Service workers are unavailable. Offline page reloads cannot be enabled in this browser.');
  const registration = await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}service-worker.js`);
  // Await this registration, not an unrelated worker on another app's scope.
  if (registration.active && navigator.serviceWorker.controller && !registration.installing && !registration.waiting) return;
  const worker = registration.installing || registration.waiting;
  if (worker) await new Promise<void>((resolve, reject) => {
    const check = () => {
      if (worker.state === 'activated') resolve();
      if (worker.state === 'redundant') reject(new Error('App caching failed. Check free storage and reload while online.'));
    };
    worker.addEventListener('statechange', check);
    check();
  });
  if (!navigator.serviceWorker.controller || (worker && navigator.serviceWorker.controller !== worker)) await new Promise<void>(resolve => {
    const check = () => {
      if (navigator.serviceWorker.controller && (!worker || navigator.serviceWorker.controller === worker)) {
        navigator.serviceWorker.removeEventListener('controllerchange', check); resolve();
      }
    };
    navigator.serviceWorker.addEventListener('controllerchange', check);
    check();
  });
}
