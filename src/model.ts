export const MODEL_ID = 'SmolLM2-360M-Instruct-q4f32_1-MLC';
export const CONTEXT_SIZE = 4096;
export const MAX_OUTPUT_TOKENS = 1024;
export type Mode = 'text' | 'image';
export type ModelOption = {
  id: string; name: string; mode: Mode; size: string; context: string; description: string;
  backend?: 'sd-turbo' | 'janus'; repo?: string; role?: string; requiresF16?: boolean;
};
export const MODELS: ModelOption[] = [
  { id: MODEL_ID, name: 'SmolLM2 360M Instruct', mode: 'text', size: '360M · 4-bit · ~194 MiB weights', context: '4,096 tokens', description: 'Smallest download. A good starting point for constrained devices.' },
  { id: 'Qwen2.5-0.5B-Instruct-q4f32_1-MLC', name: 'Qwen2.5 0.5B Instruct', mode: 'text', size: '0.5B · 4-bit · ~265 MiB weights', context: '4,096 tokens', description: 'Compact general-purpose instruction model.' },
  { id: 'Llama-3.2-1B-Instruct-q4f32_1-MLC', name: 'Llama 3.2 1B Instruct', mode: 'text', size: '1B · 4-bit · ~663 MiB weights', context: '4,096 tokens', description: 'A larger alternative for everyday tasks.' },
  { id: 'Qwen2.5-1.5B-Instruct-q4f32_1-MLC', name: 'Qwen2.5 1.5B Instruct', mode: 'text', size: '1.5B · 4-bit · ~828 MiB weights', context: '4,096 tokens', description: 'Largest text option. Allow more GPU memory and generation time.' },
  { id: 'sd-turbo', name: 'Stable Diffusion Turbo', mode: 'image', backend: 'sd-turbo', repo: 'schmuell/sd-turbo-ort-web', size: '~2.34 GiB weights', context: '77 prompt tokens · 512 × 512 pixels', description: 'One diffusion step per image. Several GB of free memory recommended. Requires GPU float16 support.', requiresF16: true },
  { id: 'janus-pro-1b', name: 'Janus Pro 1B', mode: 'image', backend: 'janus', repo: 'onnx-community/Janus-Pro-1B-ONNX', role: '<|User|>', size: '~2.1 GiB weights · mixed 4-bit / float32', context: '384 × 384 pixels · 576 image tokens', description: 'Draws an image token by token. Several GB of free memory recommended; slow generations are allowed.' },
  { id: 'janus-1.3b', name: 'Janus 1.3B', mode: 'image', backend: 'janus', repo: 'onnx-community/Janus-1.3B-ONNX', role: 'User', size: '~2.1 GiB weights · mixed 4-bit / float32', context: '384 × 384 pixels · 576 image tokens', description: 'Original Janus image model. Several GB of free memory recommended; slow generations are allowed.' },
];
export function getModel(id: string) {
  const model = MODELS.find(model => model.id === id);
  if (!model) throw new Error(`Unknown model: ${id}`);
  return model;
}
