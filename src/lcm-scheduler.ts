// Adapted from Hugging Face Diffusers v0.23.1 LCMScheduler (Apache-2.0).
// This fixed profile matches the pinned akameswa/lcm-tiny-sd-onnx-fp16 export:
// scaled-linear betas, epsilon prediction, 50 original steps, no clipping.
export const LCM_TIMESTEPS = [999, 879, 759, 639, 519, 399, 279, 159] as const;

const alphas = new Float64Array(1000);
let product = 1;
for (let t = 0; t < alphas.length; t++) {
  const rootBeta = Math.sqrt(.00085) + t / 999 * (Math.sqrt(.012) - Math.sqrt(.00085));
  product *= 1 - rootBeta * rootBeta;
  alphas[t] = product;
}

export function normalNoise(count: number): Float32Array {
  return Float32Array.from({ length: count }, () =>
    Math.sqrt(-2 * Math.log(Math.max(Math.random(), Number.EPSILON))) * Math.cos(2 * Math.PI * Math.random()));
}

export function lcmStep(sample: Float32Array, prediction: Float32Array, index: number, noise?: Float32Array): Float32Array {
  if (index < 0 || index >= LCM_TIMESTEPS.length || !Number.isInteger(index)) throw new Error('Invalid LCM step.');
  if (sample.length !== prediction.length) throw new Error('LCM output shape does not match its input.');
  const timestep = LCM_TIMESTEPS[index];
  const alpha = Math.sqrt(alphas[timestep]), beta = Math.sqrt(1 - alphas[timestep]);
  const scaledTime = timestep * 10;
  const skip = .25 / (scaledTime * scaledTime + .25);
  const out = scaledTime / Math.sqrt(scaledTime * scaledTime + .25);
  const next = LCM_TIMESTEPS[index + 1];
  const nextAlpha = next === undefined ? 1 : Math.sqrt(alphas[next]);
  const nextBeta = next === undefined ? 0 : Math.sqrt(1 - alphas[next]);
  const random = next === undefined ? undefined : noise ?? normalNoise(sample.length);
  if (random && random.length !== sample.length) throw new Error('LCM noise shape does not match its input.');
  return Float32Array.from(sample, (value, i) => {
    const denoised = out * (value - beta * prediction[i]) / alpha + skip * value;
    return nextAlpha * denoised + nextBeta * (random?.[i] ?? 0);
  });
}

// CFG is evaluated in two sequential batch-one calls to limit GPU memory.
export function guidedPrediction(conditional: Float32Array, unconditional: Float32Array, scale = 2): Float32Array {
  if (conditional.length !== unconditional.length) throw new Error('Guidance output shapes differ.');
  return Float32Array.from(conditional, (value, index) => {
    const guided = unconditional[index] + scale * (value - unconditional[index]);
    if (!Number.isFinite(guided)) throw new Error('Drawing network produced non-finite values.');
    return guided;
  });
}
