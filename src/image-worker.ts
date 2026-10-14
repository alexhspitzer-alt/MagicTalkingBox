import { loadDiffusion } from './diffusion';
import { loadTinyDiffusion } from './tiny-diffusion';
import { getModel, type ModelOption } from './model';
import { imageAssetCache } from './image-cache';
import type { ImageProgress, ImageResult, ImageGenerationOptions } from './image-inference';

type Backend = { generate: (prompt: string, report: (progress: ImageProgress) => void, options?: ImageGenerationOptions) => Promise<Blob> };
let backend: Backend | undefined;
let running = false;
self.onmessage = async (event: MessageEvent<{ id: number; type: string; model?: string; prompt?: string; runtime?: string; options?: ImageGenerationOptions }>) => {
  const { id, type } = event.data;
  const report = (progress: ImageProgress) => self.postMessage({ id, type: 'progress', progress });
  if (running) { self.postMessage({ id, type: 'error', error: 'The image worker is already busy.' }); return; }
  running = true;
  try {
    let result: ImageResult | undefined;
    if (type === 'load') {
      const option = getModel(event.data.model!);
      if (option.mode !== 'image') throw new Error('Select an image model first.');
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('WebGPU is unavailable in the image worker.');
      if (option.requiresF16 && !adapter.features.has('shader-f16')) throw new Error(`${option.name} requires GPU float16 support. Try Janus or another WebGPU device.`);
      const cache = await imageAssetCache();
      const { env } = await import('@huggingface/transformers');
      // Offline optional metadata should return null instead of fetching HF.
      env.allowLocalModels = !navigator.onLine;
      env.allowRemoteModels = navigator.onLine;
      env.useBrowserCache = false; env.useFSCache = false;
      env.useCustomCache = true; env.customCache = cache.tracked;
      if (!env.backends.onnx.wasm) throw new Error('ONNX WebAssembly runtime configuration is unavailable.');
      env.backends.onnx.wasm.wasmPaths = event.data.runtime!;
      env.backends.onnx.wasm.numThreads = 1;
      env.backends.onnx.wasm.proxy = false;
      backend = option.backend === 'sd-turbo'
        ? await loadDiffusion(option, event.data.runtime!, cache, report)
        : option.backend === 'tiny-lcm'
          ? await loadTinyDiffusion(option, event.data.runtime!, cache, report)
          : await loadJanus(option, report);
      await cache.complete(option.id);
      report({ text: option.stagedLoading ? 'Image weights cached. Networks load one at a time when drawing.' : 'Image model loaded and cached.', progress: 1 });
    } else if (type === 'generate') {
      if (!backend) throw new Error('Load an image model first.');
      const start = performance.now();
      let timings: Record<string, number> | undefined;
      const blob = await backend.generate(event.data.prompt!, progress => {
        if (progress.timings) timings = progress.timings;
        report(progress);
      }, event.data.options);
      result = { blob, seconds: (performance.now() - start) / 1000, timings, steps: event.data.options?.steps };
    } else throw new Error(`Unknown image operation: ${type}`);
    self.postMessage({ id, type: 'result', result });
  } catch (error) {
    self.postMessage({ id, type: 'error', error: `${error instanceof Error ? error.message : String(error)} If loading fails, check free storage and GPU memory; try another model.` });
  } finally { running = false; }
};

async function loadJanus(option: ModelOption, report: (progress: ImageProgress) => void): Promise<Backend> {
  const { AutoProcessor, MultiModalityCausalLM, BaseStreamer } = await import('@huggingface/transformers');
  const progress_callback = (event: { status: string; file?: string; progress?: number; loaded?: number }) => {
    report({ text: `${event.status}: ${event.file || option.name}${event.progress !== undefined ? ` · ${event.progress.toFixed(0)}%` : ''}` });
  };
  const processor = await AutoProcessor.from_pretrained(option.repo!, { progress_callback });
  // The input embedding graph uses WASM due to upstream WebGPU compatibility.
  // The autoregressive decoder and image decoder run on WebGPU. Use one stable
  // quantization profile so cached weights never depend on GPU float16 support.
  const model = await MultiModalityCausalLM.from_pretrained(option.repo!, {
    dtype: { prepare_inputs_embeds: 'q4', language_model: 'q4', lm_head: 'q4', gen_head: 'fp32', gen_img_embeds: 'fp32', image_decode: 'fp32' },
    device: { prepare_inputs_embeds: 'wasm', language_model: 'webgpu', lm_head: 'webgpu', gen_head: 'webgpu', gen_img_embeds: 'webgpu', image_decode: 'webgpu' },
    progress_callback,
  }) as InstanceType<typeof MultiModalityCausalLM>;
  return {
    async generate(prompt, report) {
      const inputs = await processor([{ role: option.role!, content: prompt }], { chat_template: 'text_to_image' });
      const total = (processor as typeof processor & { num_image_tokens: number }).num_image_tokens;
      if (!Number.isInteger(total) || total <= 0) throw new Error('The image model configuration has no valid image token count.');
      const start = performance.now();
      class ImageStreamer extends BaseStreamer {
        private first = true;
        private tokens = 0;
        put(value: bigint[][]) {
          if (this.first) { this.first = false; return; }
          this.tokens += value[0]?.length || 0;
          report({ text: `Drawing locally: ${this.tokens} / ${total} image tokens`, progress: this.tokens / total, tokens: this.tokens, tokensPerSecond: this.tokens / Math.max((performance.now() - start) / 1000, .001) });
        }
        end() { report({ text: 'Decoding image locally…', progress: 1 }); }
      }
      const images = await model.generate_images({ ...inputs, min_new_tokens: total, max_new_tokens: total, do_sample: true, streamer: new ImageStreamer() });
      if (!images[0]) throw new Error('Janus returned no image.');
      return images[0].toBlob('image/png');
    },
  };
}
