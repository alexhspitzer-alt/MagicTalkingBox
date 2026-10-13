import type { ModelOption } from './model';
import type { imageAssetCache } from './image-cache';
import type { ImageProgress } from './image-inference';
import { DiffusionSessions } from './diffusion-sessions';
import { LCM_TIMESTEPS, lcmStep, normalNoise } from './lcm-scheduler';

export const TINY_REVISION = '3b423c4e1d61d50c56efdde24592862be68c8574';
export const TINY_FILES = ['text_encoder/model.onnx', 'unet/model.onnx', 'unet/model.onnx_data', 'vae_decoder/model.onnx', 'vae_decoder/model.onnx_data'] as const;
const SIDE = 32;

export async function loadTinyDiffusion(option: ModelOption, runtime: string, cache: Awaited<ReturnType<typeof imageAssetCache>>, report: (progress: ImageProgress) => void) {
  const ort = await import('onnxruntime-web/webgpu');
  const { AutoTokenizer } = await import('@huggingface/transformers');
  ort.env.wasm.wasmPaths = runtime; ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
  const tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
  // SD 1.5 CLIP pads with EOS; SD-Turbo's zero-padding profile is different.
  tokenizer.pad_token_id = 49407;
  const url = (file: string) => `https://huggingface.co/${option.repo}/resolve/${TINY_REVISION}/${file}`;
  for (const file of TINY_FILES) {
    report({ text: `Caching Tiny SD ${file}. GPU initialization starts when drawing.` });
    await cache.ensure(url(file));
  }
  let reportStage = report;
  const sessions = new DiffusionSessions(true, async name => {
    const dimensions: Record<string, number> = name === 'unet'
      ? { unet_sample_batch: 1, unet_sample_channels: 4, unet_sample_height: SIDE, unet_sample_width: SIDE, unet_timestep_batch: 1, unet_ehs_batch: 1, unet_ehs_sequence: 77 }
      : name === 'vae_decoder'
        ? { vaedec_sample_batch: 1, vaedec_sample_channels: 4, vaedec_sample_height: SIDE, vaedec_sample_width: SIDE }
        : { textenc_inputids_batch: 1, textenc_inputids_sequence: 77 };
    reportStage({ text: `Loading Tiny SD ${name === 'unet' ? 'drawing network · ~617 MiB weights' : name === 'text_encoder' ? 'prompt encoder' : 'image decoder'} from local cache…` });
    const common: import('onnxruntime-web').InferenceSession.SessionOptions = {
      executionProviders: ['webgpu'], preferredOutputLocation: 'cpu',
      enableMemPattern: false, enableCpuMemArena: false,
      freeDimensionOverrides: dimensions,
      extra: { session: { disable_prepacking: '1', use_device_allocator_for_initializers: '1' } },
    };
    return cache.withModelURL(url(`${name}/model.onnx`), modelURL => {
      if (name === 'text_encoder') return ort.InferenceSession.create(modelURL, common);
      // The graph declares this exact relative path. Supplying it explicitly
      // prevents ORT from attempting to resolve external weights beside a Blob URL.
      return cache.withModelURL(url(`${name}/model.onnx_data`), weightsURL =>
        ort.InferenceSession.create(modelURL, { ...common, externalData: [{ path: 'model.onnx_data', data: weightsURL }] }));
    });
  });
  return {
    async generate(prompt: string, report: (progress: ImageProgress) => void): Promise<Blob> {
      reportStage = report;
      const tensors = new Set<import('onnxruntime-web').Tensor>();
      const keep = <T extends import('onnxruntime-web').Tensor>(tensor: T) => { tensors.add(tensor); return tensor; };
      const dispose = (tensor: import('onnxruntime-web').Tensor) => { tensor.dispose(); tensors.delete(tensor); };
      try {
        const tokenized = tokenizer(prompt, { padding: 'max_length', max_length: 77, truncation: true, return_tensor: false });
        const embedding = await sessions.use('text_encoder', async session => {
          const ids = keep(new ort.Tensor('int32', Int32Array.from(tokenized.input_ids), [1, 77]));
          try {
            report({ text: 'Encoding Tiny SD prompt locally…', progress: .05 });
            const outputs = await session.run({ input_ids: ids });
            Object.values(outputs).forEach(keep);
            const result = outputs.last_hidden_state;
            if (!result) throw new Error('Tiny SD prompt encoder returned no embedding.');
            for (const tensor of Object.values(outputs)) if (tensor !== result) dispose(tensor);
            return result;
          } finally { dispose(ids); }
        });
        let latent = normalNoise(4 * SIDE * SIDE);
        // Keep only the smaller UNet resident across its four steps. Guidance
        // scale 1 matches the model author's recipe: one batch, no CFG duplication.
        await sessions.use('unet', async session => {
          for (let index = 0; index < LCM_TIMESTEPS.length; index++) {
            report({ text: `Drawing Tiny SD locally: step ${index + 1} / 4…`, progress: .15 + index * .15 });
            const sample = keep(new ort.Tensor('float32', latent, [1, 4, SIDE, SIDE]));
            const timestep = keep(new ort.Tensor('float32', Float32Array.of(LCM_TIMESTEPS[index]), [1]));
            let outputs: Record<string, import('onnxruntime-web').Tensor> = {};
            try {
              outputs = await session.run({ sample, timestep, encoder_hidden_states: embedding });
              Object.values(outputs).forEach(keep);
              if (!outputs.out_sample) throw new Error('Tiny SD drawing network returned no prediction.');
              latent = lcmStep(latent, outputs.out_sample.data as Float32Array, index);
            } finally {
              Object.values(outputs).forEach(dispose); dispose(sample); dispose(timestep);
            }
          }
        });
        dispose(embedding);
        for (let i = 0; i < latent.length; i++) latent[i] /= .18215;
        const decoded = await sessions.use('vae_decoder', async session => {
          const input = keep(new ort.Tensor('float32', latent, [1, 4, SIDE, SIDE]));
          try {
            report({ text: 'Decoding Tiny SD image locally…', progress: .85 });
            const outputs = await session.run({ latent_sample: input });
            Object.values(outputs).forEach(keep);
            if (!outputs.sample) throw new Error('Tiny SD decoder returned no image.');
            return outputs.sample;
          } finally { dispose(input); }
        });
        const [, channels, height, width] = decoded.dims;
        if (channels !== 3 || height !== 256 || width !== 256) throw new Error('Tiny SD decoder returned an unexpected image shape.');
        const pixels = decoded.data as Float32Array;
        const rgba = new Uint8ClampedArray(width * height * 4);
        for (let i = 0; i < width * height; i++) {
          for (let c = 0; c < 3; c++) rgba[4 * i + c] = Math.round(Math.max(0, Math.min(1, pixels[c * width * height + i] / 2 + .5)) * 255);
          rgba[4 * i + 3] = 255;
        }
        const canvas = new OffscreenCanvas(width, height), context = canvas.getContext('2d');
        if (!context) throw new Error('Image rendering is unavailable on this device.');
        context.putImageData(new ImageData(rgba, width, height), 0, 0);
        return await canvas.convertToBlob({ type: 'image/png' });
      } finally { for (const tensor of tensors) tensor.dispose(); }
    },
  };
}
