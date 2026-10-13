import { test, expect } from '@playwright/test';
import { LCM_TIMESTEPS, lcmStep } from '../src/lcm-scheduler';

test('LCM schedule and updates match the pinned Diffusers recipe', () => {
  expect([...LCM_TIMESTEPS]).toEqual([999, 759, 519, 279]);
  // Independent double-precision reference from Diffusers v0.23.1 equations,
  // sample=1, epsilon=.2; deterministic zero added noise between steps.
  const reference = [2.6793821918439433, 1.7676655106361738, 1.3042195299055628, 1.1094952636161592];
  reference.forEach((expected, index) => {
    const result = lcmStep(Float32Array.of(1), Float32Array.of(.2), index, Float32Array.of(0));
    expect(result[0]).toBeCloseTo(expected, 5);
  });
});

test('LCM injects noise only between steps and validates tensor sizes', () => {
  const sample = Float32Array.of(1), prediction = Float32Array.of(.2);
  expect(lcmStep(sample, prediction, 0, Float32Array.of(1))[0]).toBeGreaterThan(lcmStep(sample, prediction, 0, Float32Array.of(0))[0]);
  expect(lcmStep(sample, prediction, 3, Float32Array.of(1000))[0]).toBe(lcmStep(sample, prediction, 3, Float32Array.of(0))[0]);
  expect(() => lcmStep(sample, new Float32Array(2), 0)).toThrow('shape');
  expect(() => lcmStep(sample, prediction, 4)).toThrow('step');
});
