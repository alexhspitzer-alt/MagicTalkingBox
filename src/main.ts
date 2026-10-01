import './style.css';
import { detectDevice } from './device';
import { Inference } from './inference';
import { cacheApp, modelCacheInfo, readChat, saveChat, clearChat, storageInfo, type Message } from './persistence';
import { UI, element } from './ui';
import { MODEL } from './model';

const ui = new UI();
let supported = false, loaded = false, busy = false, appCached = false, modelCached = false;
let messages: Message[] = [];
try { messages = readChat(); ui.render(messages); } catch (error) { ui.error(error); }
const inference = new Inference(error => { loaded = false; ui.error(error); controls(); });
function controls(generating = false) { ui.controls(loaded, busy, supported, generating); }
function offlineReadiness() {
  element('offline-status').textContent = appCached && modelCached
    ? 'Offline ready. You can disconnect, chat, and reload this page from this browser.'
    : 'First visit: load the model while online. Wait for both app and model caches before disconnecting.';
}
async function refreshStorage(requestPersistence = false) {
  try {
    const info = await storageInfo(requestPersistence);
    element('persistence').textContent = info.persisted ? 'Storage: persistent' : 'Storage: best effort (browser may evict)';
    element('disk').textContent = info.usage === undefined ? 'Storage size unavailable' : `Origin storage: ${(info.usage / 1048576).toFixed(0)} MiB${info.quota ? ` / ${(info.quota / 1073741824).toFixed(1)} GiB quota` : ''}`;
    const cache = await modelCacheInfo();
    modelCached = cache.complete;
    element('model-cache').textContent = cache.complete ? 'Model cache: complete' : 'Model cache: incomplete';
    if (cache.bytes) element('size').textContent = `360M parameters · 4-bit · ${(cache.bytes / 1048576).toFixed(0)} MiB weights`;
    offlineReadiness();
  } catch (error) {
    modelCached = false;
    element('model-cache').textContent = 'Model cache: could not verify';
    ui.error(error); offlineReadiness();
  }
}
ui.network();
ui.elapsed(0);
window.addEventListener('online', () => ui.network());
window.addEventListener('offline', () => ui.network());
void cacheApp().then(() => {
  appCached = !import.meta.env.DEV;
  element('app-cache').textContent = appCached ? 'App cache: complete' : 'App cache: disabled in development';
  offlineReadiness();
}).catch(error => { element('app-cache').textContent = 'App cache: failed'; ui.error(error); });
void refreshStorage();
void detectDevice().then(device => {
  supported = true;
  element('gpu').textContent = device;
  ui.status(`Ready to load. Estimated GPU allocation: ~${MODEL.vram_required_MB?.toFixed(0)} MB; actual use is not exposed by the browser.`);
  controls();
}).catch(error => { element('gpu').textContent = 'WebGPU unavailable'; ui.status('This device cannot run the model.'); ui.error(error); controls(); });

element('load').addEventListener('click', async () => {
  if (busy) return;
  busy = true; ui.clearError(); controls();
  const progress = element<HTMLProgressElement>('progress');
  progress.hidden = false; progress.value = 0;
  try {
    await refreshStorage(true);
    await inference.load(report => { ui.status(report.text); progress.value = report.progress; });
    loaded = true;
    element('load').textContent = 'Model loaded';
    ui.status('Ready. All inference runs locally.');
    await refreshStorage();
  } catch (error) {
    loaded = false;
    ui.error(error); ui.status('Model loading failed. Retry when your device and storage are ready.');
  } finally { busy = false; progress.hidden = true; controls(); }
});

element<HTMLFormElement>('chat').addEventListener('submit', async event => {
  event.preventDefault();
  const prompt = element<HTMLTextAreaElement>('prompt');
  const text = prompt.value.trim();
  if (!text || busy || !loaded) return;
  ui.clearError();
  const user: Message = { role: 'user', content: text };
  const answer: Message = { role: 'assistant', content: '' };
  const input = [...messages, user];
  // Commit the user prompt immediately; interrupted replies survive refreshes.
  messages.push(user, answer);
  persist(); ui.render(messages); prompt.value = '';
  busy = true; controls(true); ui.status('Thinking locally… Slow generations can run as long as they need.');
  element('speed').textContent = 'Measuring…';
  const start = performance.now();
  const elapsed = () => ui.elapsed((performance.now() - start) / 1000);
  elapsed();
  const timer = window.setInterval(elapsed, 250);
  let checkpoint = performance.now();
  try {
    const reason = await inference.generate(input, delta => {
      answer.content += delta; ui.append(delta);
      if (performance.now() - checkpoint > 1000) { persist(); checkpoint = performance.now(); }
    }, usage => { element('speed').textContent = `${usage.extra.decode_tokens_per_s.toFixed(2)} tokens/s · ${usage.completion_tokens} tokens`; });
    ui.status(reason === 'length' ? 'Reached the response or context limit (1,024 output / 4,096 total tokens). Continue or start a new chat as needed.' : 'Ready.');
  } catch (error) {
    loaded = false;
    element('load').textContent = 'Reload model';
    ui.error(error);
    ui.status('Generation failed. Start a new chat if the context is full; reload the model if the GPU was lost.');
  } finally {
    window.clearInterval(timer); elapsed();
    if (!answer.content) { answer.content = '[No response — generation was interrupted.]'; ui.render(messages); }
    persist(); busy = false; controls(); prompt.focus();
    if (element('speed').textContent === 'Measuring…') element('speed').textContent = 'Unavailable for this generation';
    void refreshStorage();
  }
});
function persist() { try { saveChat(messages); } catch (error) { ui.error(new Error(`Conversation could not be saved: ${String(error)}`)); } }
element('prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); element<HTMLFormElement>('chat').requestSubmit(); }
});
element('stop').addEventListener('click', () => { ui.status('Stopping after the current inference step…'); void inference.stop().catch(error => ui.error(error)); });
element('clear').addEventListener('click', () => {
  if (busy) return;
  try { clearChat(); messages = []; ui.render(messages); ui.clearError(); ui.status(loaded ? 'Ready. New conversation.' : 'Load the model to start a conversation.'); }
  catch (error) { ui.error(error); }
});
