import type { ModelOption } from './model';
import type { imageAssetCache } from './image-cache';
import type { ImageProgress } from './image-inference';
import { DiffusionSessions } from './diffusion-sessions';

// SD-Turbo's one-step Euler pipeline follows Microsoft's MIT-licensed WebGPU
// example: microsoft/onnxruntime-inference-examples/js/sd-turbo/index.js.
export async function loadDiffusion(option: ModelOption, runtime: string, cache: Awaited<ReturnType<typeof imageAssetCache>>, report: (progress: ImageProgress) => void): Promise<{ generate: (prompt: string, report: (progress: ImageProgress) => void) => Promise<Blob> }> {
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
  const side = option.stagedLoading ? 32 : 64; // 256px in low-memory mode; latent scale is 8.
  const specifications: { name: string; dimensions: Record<string, number> }[] = [
    { name: 'unet', dimensions: { batch_size: 1, num_channels: 4, height: side, width: side, sequence_length: 77 } },
    { name: 'text_encoder', dimensions: { batch_size: 1 } },
    { name: 'vae_decoder', dimensions: { batch_size: 1, num_channels_latent: 4, height_latent: side, width_latent: side } },
  ];
  const url = (name: string) => `https://huggingface.co/${option.repo}/resolve/main/${name}/model.onnx`;
  let reportStage = report;
  const label = (name: string) => name === 'unet' ? 'drawing network' : name === 'text_encoder' ? 'prompt encoder' : 'image decoder';
  const sessions = new DiffusionSessions(Boolean(option.stagedLoading), async name => {
    const spec = specifications.find(spec => spec.name === name)!;
    reportStage({ text: `Loading ${label(name)}${option.stagedLoading ? ' from local cache · one network at a time' : ''}. This may take several minutes.` });
    const options: import('onnxruntime-web').InferenceSession.SessionOptions = {
      ...common, freeDimensionOverrides: spec.dimensions,
      // CPU outputs survive session release between stages. Standard mode
      // keeps the prompt embedding on the GPU for its retained UNet session.
      preferredOutputLocation: option.stagedLoading ? 'cpu' : spec.name === 'text_encoder' ? { last_hidden_state: 'gpu-buffer' } : 'cpu',
    };
    // ORT's URL loader copies into WASM before creating the session, so its
    // temporary source buffer can be collected during graph initialization.
    // Passing our own Uint8Array retains that large buffer throughout create().
    return cache.withModelURL(url(name), modelURL => ort.InferenceSession.create(modelURL, options));
  });
  if (option.stagedLoading) {
    for (const spec of specifications) {
      report({ text: `Caching ${label(spec.name)} without loading it into GPU memory. Large files may take several minutes.` });
      await cache.ensure(url(spec.name));
    }
  }
  await sessions.preload(specifications.map(spec => spec.name));
  return {
    async generate(prompt, report) {
      reportStage = report;
      const tensors = new Set<import('onnxruntime-web').Tensor>();
      const keep = <T extends import('onnxruntime-web').Tensor>(tensor: T) => { tensors.add(tensor); return tensor; };
      const collect = (outputs: Record<string, import('onnxruntime-web').Tensor>) => { Object.values(outputs).forEach(keep); return outputs; };
      try {
        report({ text: 'Encoding image prompt locally…', progress: .05 });
        const tokenized = tokenizer(prompt, { padding: 'max_length', max_length: 77, truncation: true, return_tensor: false });
        const ids = keep(new ort.Tensor('int32', Int32Array.from(tokenized.input_ids), [1, tokenized.input_ids.length]));
        const { last_hidden_state } = collect(await sessions.use('text_encoder', session => session.run({ input_ids: ids })));
        const sigma = 14.6146, count = 4 * side * side;
        const noise = new Float32Array(count), scaled = new Float32Array(count);
        for (let i = 0; i < count; i++) {
          noise[i] = Math.sqrt(-2 * Math.log(Math.max(Math.random(), Number.EPSILON))) * Math.cos(2 * Math.PI * Math.random()) * sigma;
          scaled[i] = noise[i] / Math.sqrt(sigma * sigma + 1);
        }
        report({ text: 'Drawing locally: diffusion step 1 / 1…', progress: .2 });
        const outputs = collect(await sessions.use('unet', session => session.run({
          sample: keep(new ort.Tensor('float32', scaled, [1, 4, side, side])),
          timestep: keep(new ort.Tensor('int64', BigInt64Array.from([999n]), [1])), encoder_hidden_states: last_hidden_state,
        })));
        const prediction = outputs.out_sample.data as Float32Array;
        const latent = new Float32Array(count);
        for (let i = 0; i < count; i++) latent[i] = (noise[i] - sigma * prediction[i]) / .18215;
        report({ text: 'Decoding image locally…', progress: .8 });
        const { sample } = collect(await sessions.use('vae_decoder', session => session.run({ latent_sample: keep(new ort.Tensor('float32', latent, [1, 4, side, side])) })));
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
