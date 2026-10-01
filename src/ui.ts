import { MODEL_NAME } from './model';
import { memoryInfo } from './device';
import type { Message } from './persistence';
export const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
export class UI {
  constructor() { element('model').textContent = MODEL_NAME; }
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
    element('stop').hidden = !generating;
    element('messages').setAttribute('aria-busy', String(generating));
  }
  render(messages: Message[]) {
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
