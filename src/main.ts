import './style.css';
import { detectDevice } from './device';
import { Inference } from './inference';
import { ImageInference } from './image-inference';
import { imageCacheInfo } from './image-cache';
import { readImages, saveImage, clearImages, type SavedImage } from './image-history';
import { cacheApp, modelCacheInfo, readChat, saveChat, clearChat, storageInfo, type Message } from './persistence';
import { UI, element } from './ui';
import { textModel } from './text-model';
import { MODELS, MODEL_ID, getModel, type Mode } from './model';

const ui = new UI();
let supported = false, loaded = false, busy = false, appCached = false, modelCached = false;
let mode: Mode = 'text', selected = MODEL_ID, selectionVersion = 0, stopped = false;
let messages: Message[] = [], images: SavedImage[] = [];
const remembered: Record<Mode, string> = { text: MODEL_ID, image: 'sd-turbo' };
try {
  const saved = JSON.parse(localStorage.getItem('magic-box.selection.v1') || 'null');
  if (saved) {
    for (const candidate of ['text', 'image'] as const) {
      if (MODELS.some(model => model.mode === candidate && model.id === saved[candidate])) remembered[candidate] = saved[candidate];
    }
    if (saved.mode === 'image') mode = 'image';
  }
  selected = remembered[mode]; messages = readChat();
} catch (error) { ui.error(error); }
const inference = new Inference(error => { loaded = false; ui.error(error); controls(); });
const imageInference = new ImageInference();
function controls(generating = false) { ui.controls(loaded, busy, supported, generating); }
function offlineReadiness() {
  element('offline-status').textContent = appCached && modelCached
    ? 'Offline ready. You can disconnect, generate, and reload this page with the selected model from this browser.'
    : 'Load the selected model while online. Each model downloads separately; wait for both app and model caches before disconnecting.';
}
async function refreshStorage(requestPersistence = false) {
  const id = selected, version = selectionVersion;
  try {
    const info = await storageInfo(requestPersistence);
    element('persistence').textContent = info.persisted ? 'Storage: persistent' : 'Storage: best effort (browser may evict)';
    element('disk').textContent = info.usage === undefined ? 'Storage size unavailable' : `Origin storage: ${(info.usage / 1048576).toFixed(0)} MiB${info.quota ? ` / ${(info.quota / 1073741824).toFixed(1)} GiB quota` : ''}`;
    const cache = getModel(id).mode === 'text' ? await modelCacheInfo(id) : await imageCacheInfo(id);
    if (version !== selectionVersion) return;
    modelCached = cache.complete;
    element('model-cache').textContent = cache.complete ? 'Model cache: complete' : 'Model cache: incomplete';
    if (cache.bytes) element('size').textContent = `${getModel(id).size.replace(/ · ~[^·]+ weights$/, '')} · ${(cache.bytes / 1048576).toFixed(0)} MiB cached`;
    offlineReadiness();
  } catch (error) {
    if (version !== selectionVersion) return;
    modelCached = false;
    element('model-cache').textContent = 'Model cache: could not verify';
    ui.error(error); offlineReadiness();
  }
}
async function renderHistory() {
  if (mode === 'text') { ui.render(messages); return; }
  const version = selectionVersion;
  try { images = await readImages(); if (version === selectionVersion) ui.renderImages(images); }
  catch (error) { ui.error(new Error(`Image history could not be read: ${String(error)}`)); }
}
function updateSelection() {
  selectionVersion++;
  const model = getModel(selected);
  element<HTMLSelectElement>('mode-select').value = mode;
  element<HTMLSelectElement>('model-select').replaceChildren(...MODELS.filter(option => option.mode === mode).map(option => {
    const item = document.createElement('option'); item.value = option.id; item.textContent = option.name; return item;
  }));
  element<HTMLSelectElement>('model-select').value = selected;
  element('model').textContent = model.name;
  element('size').textContent = model.size;
  element('context').textContent = model.context;
  element('model-description').textContent = model.description;
  element('terminal-title').textContent = mode === 'text' ? 'CONVERSATION / LOCAL' : 'IMAGE GALLERY / LOCAL';
  element('clear').textContent = mode === 'text' ? 'New chat' : 'Clear gallery';
  element('send').textContent = mode === 'text' ? 'Send ↗' : 'Draw ↗';
  element<HTMLTextAreaElement>('prompt').placeholder = mode === 'text' ? 'Talk to the box…' : 'Describe the image you want to draw…';
  element<HTMLTextAreaElement>('prompt').value = '';
  element('load').textContent = 'Load model'; element('speed').textContent = '—'; ui.elapsed(0);
  modelCached = false; element('model-cache').textContent = 'Model cache: checking…'; offlineReadiness();
  void renderHistory(); controls(); void refreshStorage();
}
function select(nextMode: Mode, id: string) {
  if (busy) return;
  inference.unload(); imageInference.unload(); loaded = false;
  mode = nextMode; selected = id; remembered[mode] = id;
  ui.clearError(); updateSelection();
  try { localStorage.setItem('magic-box.selection.v1', JSON.stringify({ mode, ...remembered })); } catch (error) { ui.error(error); }
  ui.status(supported ? 'Model selected. Load it to begin; cached models can load offline.' : 'This device cannot run WebGPU models.');
}
element<HTMLSelectElement>('mode-select').addEventListener('change', event => {
  const next = (event.target as HTMLSelectElement).value as Mode; select(next, remembered[next]);
});
element<HTMLSelectElement>('model-select').addEventListener('change', event => select(mode, (event.target as HTMLSelectElement).value));
updateSelection(); ui.network();
window.addEventListener('online', () => ui.network()); window.addEventListener('offline', () => ui.network());
void cacheApp().then(() => {
  appCached = !import.meta.env.DEV;
  element('app-cache').textContent = appCached ? 'App cache: complete' : 'App cache: disabled in development';
  offlineReadiness();
}).catch(error => { element('app-cache').textContent = 'App cache: failed'; ui.error(error); });
void detectDevice().then(device => {
  supported = true; element('gpu').textContent = device;
  ui.status('Ready to load a model. All generation runs on this device.'); controls();
}).catch(error => { element('gpu').textContent = 'WebGPU unavailable'; ui.status('This device cannot run WebGPU models.'); ui.error(error); controls(); });

element('load').addEventListener('click', async () => {
  if (busy) return;
  busy = true; ui.clearError(); controls();
  const progress = element<HTMLProgressElement>('progress'); progress.hidden = false; progress.removeAttribute('value');
  try {
    await refreshStorage(true);
    if (mode === 'text') {
      const allocation = textModel(selected).vram_required_MB;
      ui.status(`Loading ${getModel(selected).name}. Estimated GPU allocation: ~${allocation?.toFixed(0)} MB.`);
      await inference.load(selected, report => { ui.status(report.text); progress.value = report.progress; });
    } else {
      await imageInference.load(selected, report => {
        ui.status(report.text);
        if (report.progress === undefined) progress.removeAttribute('value'); else progress.value = report.progress;
      });
    }
    loaded = true; element('load').textContent = 'Model loaded'; ui.status('Ready. All inference runs locally.');
    await refreshStorage();
  } catch (error) {
    loaded = false; inference.unload(); imageInference.unload();
    ui.error(error); ui.status('Model loading failed. Check free memory and storage, then retry or select another model.');
  } finally { busy = false; progress.hidden = true; controls(); }
});

element<HTMLFormElement>('chat').addEventListener('submit', async event => {
  event.preventDefault();
  const prompt = element<HTMLTextAreaElement>('prompt'); const text = prompt.value.trim();
  if (!text || busy || !loaded) return;
  ui.clearError(); stopped = false;
  busy = true; controls(true); ui.status('Generating locally… Slow generations can run as long as they need.');
  element('speed').textContent = 'Measuring…'; prompt.value = '';
  const start = performance.now(), elapsed = () => ui.elapsed((performance.now() - start) / 1000);
  elapsed(); const timer = window.setInterval(elapsed, 250);
  let answer: Message | undefined;
  try {
    if (mode === 'image') {
      const result = await imageInference.generate(text, report => {
        ui.status(report.text);
        if (report.tokensPerSecond !== undefined) element('speed').textContent = `${report.tokensPerSecond.toFixed(2)} image tokens/s · ${report.tokens} tokens`;
      });
      const image: SavedImage = { id: crypto.randomUUID(), prompt: text, model: getModel(selected).name, blob: result.blob, seconds: result.seconds, created: Date.now() };
      images.push(image); ui.renderImages(images);
      element('speed').textContent = `${(60 / Math.max(result.seconds, .001)).toFixed(2)} images/min · ${result.seconds.toFixed(1)} s/image`;
      try { await saveImage(image); ui.status('Ready. Image saved on this device.'); }
      catch (error) { ui.error(new Error(`Image generated, but could not be saved: ${String(error)}. Download the PNG before closing this page.`)); ui.status('Image generated. Download it to keep a copy.'); }
    } else {
      const user: Message = { role: 'user', content: text };
      answer = { role: 'assistant', content: '' }; const input = [...messages, user];
      messages.push(user, answer); persist(); ui.render(messages);
      let checkpoint = performance.now();
      const reason = await inference.generate(input, delta => {
        answer!.content += delta; ui.append(delta);
        if (performance.now() - checkpoint > 1000) { persist(); checkpoint = performance.now(); }
      }, usage => { element('speed').textContent = `${usage.extra.decode_tokens_per_s.toFixed(2)} tokens/s · ${usage.completion_tokens} tokens`; });
      ui.status(reason === 'length' ? 'Reached the response or context limit (1,024 output / 4,096 total tokens). Continue or start a new chat as needed.' : 'Ready.');
    }
  } catch (error) {
    if (stopped && mode === 'image') ui.status('Drawing stopped. Reload the cached model to draw again.');
    else { ui.error(error); ui.status('Generation failed. Reload the model or choose a smaller model if memory is limited.'); }
    loaded = false; element('load').textContent = 'Reload model'; inference.unload(); imageInference.unload();
  } finally {
    window.clearInterval(timer); elapsed();
    if (answer) {
      if (!answer.content) { answer.content = '[No response — generation was interrupted.]'; ui.render(messages); }
      persist();
    }
    busy = false; controls(); prompt.focus();
    if (element('speed').textContent === 'Measuring…') element('speed').textContent = 'Unavailable for this generation';
    void refreshStorage();
  }
});
function persist() { try { saveChat(messages); } catch (error) { ui.error(new Error(`Conversation could not be saved: ${String(error)}`)); } }
element('prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); element<HTMLFormElement>('chat').requestSubmit(); }
});
element('stop').addEventListener('click', () => {
  stopped = true;
  if (mode === 'image') { loaded = false; imageInference.unload(); }
  else { ui.status('Stopping after the current inference step…'); void inference.stop().catch(error => ui.error(error)); }
});
element('clear').addEventListener('click', async () => {
  if (busy) return;
  busy = true; controls();
  try {
    if (mode === 'image') { await clearImages(); images = []; ui.renderImages(images); }
    else { clearChat(); messages = []; ui.render(messages); }
    ui.clearError(); ui.status(loaded ? 'Ready.' : 'Load the selected model to begin.');
  } catch (error) { ui.error(error); }
  finally { busy = false; controls(); }
});
