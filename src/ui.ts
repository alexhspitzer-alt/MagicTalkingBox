import { memoryInfo } from './device';
import type { Message } from './persistence';
import type { SavedImage } from './image-history';
export const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
export class UI {
  private imageURLs: string[] = [];
  status(text: string) { element('status').textContent = text; }
  error(error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    element('error').hidden = false;
    element('error').textContent = message + (navigator.onLine ? '' : ' You are offline. If any model files are missing, reconnect to finish downloading.');
  }
  clearError() { element('error').hidden = true; element('error').textContent = ''; }
  controls(loaded: boolean, busy: boolean, supported: boolean, generating = false) {
    element<HTMLButtonElement>('load').disabled = busy || !supported || loaded;
    element<HTMLButtonElement>('send').disabled = busy || !loaded;
    element<HTMLTextAreaElement>('prompt').disabled = busy || !loaded;
    element<HTMLButtonElement>('clear').disabled = busy;
    element<HTMLSelectElement>('mode-select').disabled = busy;
    element<HTMLSelectElement>('model-select').disabled = busy;
    element('stop').hidden = !generating;
    element('messages').setAttribute('aria-busy', String(generating));
  }
  render(messages: Message[]) {
    this.releaseImages();
    if (!messages.length) {
      element('messages').innerHTML = '<p class="empty">The box is quiet.<br><span>Load the model, then give it something to think about.</span></p>';
      return;
    }
    element('messages').replaceChildren(...messages.map(message => {
      const article = document.createElement('article');
      article.className = `message ${message.role}`;
      const label = document.createElement('span');
      label.className = 'message-label'; label.textContent = message.role === 'user' ? 'YOU' : 'BOX';
      const content = document.createElement('div');
      content.className = 'message-content'; content.textContent = message.content;
      article.append(label, content); return article;
    }));
    this.scroll();
  }
  renderImages(images: SavedImage[]) {
    this.releaseImages();
    if (!images.length) {
      const empty = document.createElement('p'); empty.className = 'empty';
      empty.textContent = 'Your local gallery is empty. Load an image model, then describe a drawing.';
      element('messages').replaceChildren(empty); return;
    }
    element('messages').replaceChildren(...images.map(image => {
      const article = document.createElement('article'); article.className = 'image-result';
      const caption = document.createElement('p'); caption.textContent = image.prompt;
      const picture = document.createElement('img');
      const url = URL.createObjectURL(image.blob); this.imageURLs.push(url);
      picture.src = url; picture.alt = image.prompt; picture.loading = 'lazy';
      const details = document.createElement('p'); details.className = 'hint';
      details.textContent = `${image.model} · ${image.seconds.toFixed(1)} s · ${new Date(image.created).toLocaleString()}`;
      const download = document.createElement('a'); download.className = 'download';
      download.href = url; download.download = `magic-box-${image.id}.png`; download.textContent = 'Download PNG ↓';
      article.append(caption, picture, details, download); return article;
    }));
    this.scroll();
  }
  private releaseImages() { this.imageURLs.forEach(url => URL.revokeObjectURL(url)); this.imageURLs = []; }
  append(text: string) {
    const messages = element('messages');
    const follow = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 80;
    messages.querySelector('.message:last-child .message-content')!.textContent += text;
    if (follow) this.scroll();
  }
  private scroll() { const messages = element('messages'); messages.scrollTop = messages.scrollHeight; }
  elapsed(seconds: number) { element('elapsed').textContent = `${seconds.toFixed(1)} s`; element('memory').textContent = memoryInfo(); }
  network() { element('network').textContent = navigator.onLine ? 'Online' : 'Offline · inference stays local'; }
}
