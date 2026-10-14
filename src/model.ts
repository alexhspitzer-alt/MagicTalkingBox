export const MODEL_ID = 'SmolLM2-360M-Instruct-q4f32_1-MLC';
export const CONTEXT_SIZE = 4096;
export const MAX_OUTPUT_TOKENS = 1024;
export type Mode = 'text' | 'image';
export type ModelOption = {
  id: string; name: string; mode: Mode; size: string; context: string; description: string;
  backend?: 'sd-turbo' | 'tiny-lcm' | 'janus'; repo?: string; role?: string; requiresF16?: boolean;
  contextTokens?: number; maxOutputTokens?: number;
  stagedLoading?: boolean;
};
export const MODELS: ModelOption[] = [
  { id: MODEL_ID, name: 'SmolLM2 360M Instruct', mode: 'text', size: '360M · 4-bit · ~194 MiB weights', context: '4,096 tokens', description: 'Smallest download. A good starting point for constrained devices.' },
  { id: 'Qwen2.5-0.5B-Instruct-q4f32_1-MLC', name: 'Qwen2.5 0.5B Instruct', mode: 'text', size: '0.5B · 4-bit · ~265 MiB weights', context: '4,096 tokens', description: 'Compact general-purpose instruction model.' },
  { id: 'Llama-3.2-1B-Instruct-q4f32_1-MLC', name: 'Llama 3.2 1B Instruct', mode: 'text', size: '1B · 4-bit · ~663 MiB weights', context: '4,096 tokens', description: 'A larger alternative for everyday tasks.' },
  { id: 'Qwen2.5-1.5B-Instruct-q4f32_1-MLC', name: 'Qwen2.5 1.5B Instruct', mode: 'text', size: '1.5B · 4-bit · ~828 MiB weights', context: '4,096 tokens', description: 'A medium text option for everyday tasks.' },
  { id: 'Qwen2.5-3B-Instruct-q4f32_1-MLC', name: 'Qwen2.5 3B Instruct', mode: 'text', size: '3B · 4-bit · ~1,656 MiB weights', context: '4,096 tokens', description: 'A step up in text capability. Estimated GPU allocation: ~2,894 MB, plus browser overhead. Try this before the 7B model.' },
  { id: 'Llama-3.2-3B-Instruct-q4f32_1-MLC', name: 'Llama 3.2 3B Instruct', mode: 'text', size: '3B · 4-bit · ~1,724 MiB weights', context: '4,096 tokens', description: 'An alternative larger instruct model. Estimated GPU allocation: ~2,952 MB, plus browser overhead.' },
  { id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC', name: 'Llama 3.2 3B · lower memory', mode: 'text', size: '3B · 4-bit · ~1,724 MiB weights', context: '2,048 tokens · up to 512 output tokens', contextTokens: 2048, maxOutputTokens: 512, requiresF16: true, description: 'Uses 16-bit computation and a shorter context to reduce memory. Published GPU estimate at 4,096 tokens: ~2,264 MB; this option uses 2,048 tokens. Browser overhead is additional. Requires GPU float16 support and a separate download.' },
  { id: 'Qwen2.5-7B-Instruct-q4f32_1-MLC', name: 'Qwen2.5 7B Instruct', mode: 'text', size: '7B · 4-bit · ~4,086 MiB weights', context: '4,096 tokens', description: 'Largest text option. Estimated GPU allocation: ~5,900 MB, plus browser overhead; may exceed phone memory. Slow generations are allowed to finish.' },
  { id: 'tiny-sd-lcm', name: 'Tiny SD LCM · smaller memory', mode: 'image', backend: 'tiny-lcm', repo: 'akameswa/lcm-tiny-sd-onnx-fp16', size: '~947 MiB weights · ~617 MiB drawing network', context: '77 prompt tokens · 256 × 256 pixels · 1–50 guided steps · default 8', stagedLoading: true, requiresF16: true, description: 'A genuinely smaller drawing network: about 63% smaller than SD-Turbo. Loads one network at a time. Separate download of about 947 MiB; old SD-Turbo files stay cached. Adjustable 1–50 drawing steps (default 8) with guidance scale 2. Guidance passes run sequentially to limit memory; slower than the original four-step mode. Lower quality than larger models; memory limits can still apply. Loading caches files; GPU initialization starts when drawing.' },
  { id: 'sd-turbo', name: 'Stable Diffusion Turbo', mode: 'image', backend: 'sd-turbo', repo: 'schmuell/sd-turbo-ort-web', size: '~2.34 GiB weights', context: '77 prompt tokens · 512 × 512 pixels', description: 'One diffusion step per image. Several GB of free memory recommended. Requires GPU float16 support.', requiresF16: true },
  { id: 'sd-turbo-low-memory', name: 'Stable Diffusion Turbo · low memory', mode: 'image', backend: 'sd-turbo', repo: 'schmuell/sd-turbo-ort-web', size: '~2.34 GiB weights · shared SD-Turbo cache', context: '77 prompt tokens · 256 × 256 pixels', stagedLoading: true, requiresF16: true, description: 'Loads one network at a time and releases it before the next stage. Draws at 256 × 256 to reduce working memory; shares weights with SD-Turbo. Loading only caches files; GPU initialization starts when drawing. Slower: networks reload from the local cache for every drawing. Several GB of free memory may still be needed. Requires GPU float16 support.' },
  { id: 'janus-pro-1b', name: 'Janus Pro 1B', mode: 'image', backend: 'janus', repo: 'onnx-community/Janus-Pro-1B-ONNX', role: '<|User|>', size: '~2.1 GiB weights · mixed 4-bit / float32', context: '384 × 384 pixels · 576 image tokens', description: 'Draws an image token by token. Several GB of free memory recommended; slow generations are allowed.' },
  { id: 'janus-1.3b', name: 'Janus 1.3B', mode: 'image', backend: 'janus', repo: 'onnx-community/Janus-1.3B-ONNX', role: 'User', size: '~2.1 GiB weights · mixed 4-bit / float32', context: '384 × 384 pixels · 576 image tokens', description: 'Original Janus image model. Several GB of free memory recommended; slow generations are allowed.' },
];
export function getModel(id: string) {
  const model = MODELS.find(model => model.id === id);
  if (!model) throw new Error(`Unknown model: ${id}`);
  return model;
}
