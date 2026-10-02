import { getModel, type ModelOption } from './model';
import { imageAssetCache } from './image-cache';
import type { ImageProgress, ImageResult } from './image-inference';

type Backend = { generate: (prompt: string, report: (progress: ImageProgress) => void) => Promise<Blob> };
let backend: Backend | undefined;
let running = false;
self.onmessage = async (event: MessageEvent<{ id: number; type: string; model?: string; prompt?: string; runtime?: string }>) => {
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
        : await loadJanus(option, report);
      await cache.complete(option.id);
      report({ text: 'Image model loaded and cached.', progress: 1 });
    } else if (type === 'generate') {
      if (!backend) throw new Error('Load an image model first.');
      const start = performance.now();
      const blob = await backend.generate(event.data.prompt!, report);
      result = { blob, seconds: (performance.now() - start) / 1000 };
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

// SD-Turbo's one-step Euler pipeline follows Microsoft's MIT-licensed WebGPU
// example: microsoft/onnxruntime-inference-examples/js/sd-turbo/index.js.
async function loadDiffusion(option: ModelOption, runtime: string, cache: Awaited<ReturnType<typeof imageAssetCache>>, report: (progress: ImageProgress) => void): Promise<Backend> {
  const ort = await import('onnxruntime-web/webgpu');
  const { AutoTokenizer } = await import('@huggingface/transformers');
  ort.env.wasm.wasmPaths = runtime; ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
  report({ text: 'Loading and caching image tokenizer…' });
  const tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
  tokenizer.pad_token_id = 0;
  const common: import('onnxruntime-web').InferenceSession.SessionOptions = {
    executionProviders: ['webgpu'], enableMemPattern: false, enableCpuMemArena: false,
    extra: { session: { disable_prepacking: '1', use_device_allocator_for_initializers: '1', use_ort_model_bytes_directly: '1', use_ort_model_bytes_for_initializers: '1' } },
  };
  const specifications: { name: string; dimensions: Record<string, number> }[] = [
    { name: 'unet', dimensions: { batch_size: 1, num_channels: 4, height: 64, width: 64, sequence_length: 77 } },
    { name: 'text_encoder', dimensions: { batch_size: 1 } },
    { name: 'vae_decoder', dimensions: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } },
  ];
  const sessions: Record<string, import('onnxruntime-web').InferenceSession> = {};
  for (const spec of specifications) {
    report({ text: `Downloading / loading ${spec.name}. Large files may take several minutes.` });
    const bytes = await cache.download(`https://huggingface.co/${option.repo}/resolve/main/${spec.name}/model.onnx`);
    sessions[spec.name] = await ort.InferenceSession.create(new Uint8Array(bytes), {
      ...common, freeDimensionOverrides: spec.dimensions,
      ...(spec.name === 'text_encoder' ? { preferredOutputLocation: { last_hidden_state: 'gpu-buffer' as const } } : {}),
    });
  }
  return {
    async generate(prompt, report) {
      const tensors = new Set<import('onnxruntime-web').Tensor>();
      const keep = <T extends import('onnxruntime-web').Tensor>(tensor: T) => { tensors.add(tensor); return tensor; };
      const collect = (outputs: Record<string, import('onnxruntime-web').Tensor>) => { Object.values(outputs).forEach(keep); return outputs; };
      try {
        report({ text: 'Encoding image prompt locally…', progress: .05 });
        const tokenized = tokenizer(prompt, { padding: 'max_length', max_length: 77, truncation: true, return_tensor: false });
        const ids = keep(new ort.Tensor('int32', Int32Array.from(tokenized.input_ids), [1, tokenized.input_ids.length]));
        const { last_hidden_state } = collect(await sessions.text_encoder.run({ input_ids: ids }));
        const sigma = 14.6146, count = 4 * 64 * 64;
        const noise = new Float32Array(count), scaled = new Float32Array(count);
        for (let i = 0; i < count; i++) {
          noise[i] = Math.sqrt(-2 * Math.log(Math.max(Math.random(), Number.EPSILON))) * Math.cos(2 * Math.PI * Math.random()) * sigma;
          scaled[i] = noise[i] / Math.sqrt(sigma * sigma + 1);
        }
        report({ text: 'Drawing locally: diffusion step 1 / 1…', progress: .2 });
        const outputs = collect(await sessions.unet.run({
          sample: keep(new ort.Tensor('float32', scaled, [1, 4, 64, 64])),
          timestep: keep(new ort.Tensor('int64', BigInt64Array.from([999n]), [1])), encoder_hidden_states: last_hidden_state,
        }));
        const prediction = outputs.out_sample.data as Float32Array;
        const latent = new Float32Array(count);
        for (let i = 0; i < count; i++) latent[i] = (noise[i] - sigma * prediction[i]) / .18215;
        report({ text: 'Decoding image locally…', progress: .8 });
        const { sample } = collect(await sessions.vae_decoder.run({ latent_sample: keep(new ort.Tensor('float32', latent, [1, 4, 64, 64])) }));
        const [, , height, width] = sample.dims;
        const pixels = sample.data as Float32Array, rgba = new Uint8ClampedArray(width * height * 4);
        for (let i = 0; i < width * height; i++) {
          for (let c = 0; c < 3; c++) rgba[4 * i + c] = Math.round(Math.max(0, Math.min(1, pixels[c * width * height + i] / 2 + .5)) * 255);
          rgba[4 * i + 3] = 255;
        }
        const canvas = new OffscreenCanvas(width, height);
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Image rendering is unavailable on this device.');
        context.putImageData(new ImageData(rgba, width, height), 0, 0);
        return await canvas.convertToBlob({ type: 'image/png' });
      } finally { for (const tensor of tensors) tensor.dispose(); }
    },
  };
}
