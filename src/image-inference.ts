export type ImageProgress = { text: string; progress?: number; tokens?: number; tokensPerSecond?: number };
export type ImageResult = { blob: Blob; seconds: number };
type Reply = { id: number; type: 'progress'; progress: ImageProgress } | { id: number; type: 'result'; result: ImageResult | undefined } | { id: number; type: 'error'; error: string };
export class ImageInference {
  private worker?: Worker;
  private sequence = 0;
  private pending?: { id: number; resolve: (value: ImageResult | undefined) => void; reject: (error: Error) => void; progress: (report: ImageProgress) => void };
  unload() {
    this.worker?.terminate(); this.worker = undefined;
    this.pending?.reject(new Error('Image generation stopped. Reload the cached model to continue.'));
    this.pending = undefined;
  }
  async load(model: string, onProgress: (report: ImageProgress) => void) {
    this.unload();
    this.worker = new Worker(new URL('./image-worker.ts', import.meta.url), { type: 'module' });
    const fail = () => {
      this.pending?.reject(new Error('The image worker stopped unexpectedly. The GPU may have run out of memory. Try a different model or free device memory.'));
      this.pending = undefined;
    };
    this.worker.addEventListener('error', fail);
    this.worker.addEventListener('messageerror', fail);
    this.worker.addEventListener('message', (event: MessageEvent<Reply>) => {
      const reply = event.data, pending = this.pending;
      if (!pending || pending.id !== reply.id) return;
      if (reply.type === 'progress') pending.progress(reply.progress);
      else {
        this.pending = undefined;
        if (reply.type === 'error') pending.reject(new Error(reply.error));
        else pending.resolve(reply.result);
      }
    });
    await this.request('load', { model, runtime: new URL(import.meta.env.BASE_URL + 'ort/', location.href).href }, onProgress);
  }
  async generate(prompt: string, onProgress: (report: ImageProgress) => void): Promise<ImageResult> {
    const result = await this.request('generate', { prompt }, onProgress);
    if (!result?.blob) throw new Error('The image model returned no image.');
    return result;
  }
  private request(type: string, payload: object, progress: (report: ImageProgress) => void) {
    if (!this.worker) throw new Error('Load an image model first.');
    if (this.pending) throw new Error('The image worker is already busy.');
    const id = ++this.sequence;
    // No wall-clock timeout: slow local generations may take minutes.
    return new Promise<ImageResult | undefined>((resolve, reject) => {
      this.pending = { id, resolve, reject, progress };
      this.worker!.postMessage({ id, type, ...payload });
    });
  }
}
