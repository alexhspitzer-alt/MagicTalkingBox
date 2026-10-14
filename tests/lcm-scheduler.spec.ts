import { test, expect } from '@playwright/test';
import { LCM_TIMESTEPS, lcmTimesteps, lcmStep, guidedPrediction } from '../src/lcm-scheduler';

test('LCM schedule and updates match the pinned Diffusers recipe', () => {
  expect([...LCM_TIMESTEPS]).toEqual([999, 879, 759, 639, 519, 399, 279, 159]);
  // Independent double-precision reference from Diffusers v0.23.1 equations,
  // sample=1, epsilon=.2; deterministic zero added noise between steps.
  const reference = [1.5527020155749975, 1.3835395187163322, 1.2505325041373565, 1.149220573407876, 1.0762328383277366, 1.0282271160046927, 1.0016168434986492, 1.0124134532784144];
  reference.forEach((expected, index) => {
    const result = lcmStep(Float32Array.of(1), Float32Array.of(.2), index, Float32Array.of(0));
    expect(result[0]).toBeCloseTo(expected, 5);
  });
});

test('LCM injects noise only between steps and validates tensor sizes', () => {
  const sample = Float32Array.of(1), prediction = Float32Array.of(.2);
  expect(lcmStep(sample, prediction, 0, Float32Array.of(1))[0]).toBeGreaterThan(lcmStep(sample, prediction, 0, Float32Array.of(0))[0]);
  expect(lcmStep(sample, prediction, 7, Float32Array.of(1000))[0]).toBe(lcmStep(sample, prediction, 7, Float32Array.of(0))[0]);
  expect(() => lcmStep(sample, new Float32Array(2), 0)).toThrow('shape');
  expect(() => lcmStep(sample, prediction, 8)).toThrow('step');
});

test('sequential CFG combines predictions and rejects numerical corruption', () => {
  expect(Array.from(guidedPrediction(Float32Array.of(3, -2), Float32Array.of(1, 2)))).toEqual([5, -6]);
  expect(() => guidedPrediction(Float32Array.of(NaN), Float32Array.of(1))).toThrow('non-finite');
  expect(() => guidedPrediction(Float32Array.of(1), new Float32Array(2))).toThrow('shapes');
});

test('variable step counts retain the eight-step schedule and final-step behavior', () => {
  expect(lcmTimesteps(8)).toEqual(LCM_TIMESTEPS);
  expect(lcmTimesteps(1)).toEqual([999]);
  expect(lcmTimesteps(4)).toEqual([999, 759, 519, 279]);
  expect(lcmTimesteps(50)).toHaveLength(50);
  expect(lcmTimesteps(50).at(-1)).toBe(19);
  for (const count of [1, 4, 8, 13, 50]) {
    const schedule = lcmTimesteps(count);
    expect(new Set(schedule).size).toBe(count);
    expect(schedule.every((t, i) => i === 0 || t < schedule[i - 1])).toBe(true);
    const sample = Float32Array.of(1), prediction = Float32Array.of(.2);
    expect(lcmStep(sample, prediction, count - 1, Float32Array.of(1000), schedule)).toEqual(lcmStep(sample, prediction, count - 1, Float32Array.of(0), schedule));
  }
  for (const count of [0, 51, 1.5, NaN]) expect(() => lcmTimesteps(count)).toThrow('whole number');
});
