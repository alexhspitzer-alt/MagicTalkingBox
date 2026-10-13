import { test, expect } from '@playwright/test';
import { DiffusionSessions } from '../src/diffusion-sessions';

test('staged drawing keeps one network resident and reloads for the next image', async () => {
  let resident = 0, peak = 0;
  const events: string[] = [];
  const sessions = new DiffusionSessions(true, async name => {
    events.push(`load:${name}`); peak = Math.max(peak, ++resident);
    return { async release() { await Promise.resolve(); resident--; events.push(`release:${name}`); } };
  });
  await sessions.preload(['text_encoder', 'unet', 'vae_decoder']);
  expect(events).toEqual([]);
  for (let image = 0; image < 2; image++) {
    for (const name of ['text_encoder', 'unet', 'vae_decoder']) {
      expect(await sessions.use(name, async () => { events.push(`run:${name}`); return name; })).toBe(name);
      expect(resident).toBe(0);
    }
  }
  expect(peak).toBe(1);
  expect(events).toEqual(Array.from({ length: 2 }, () => ['text_encoder', 'unet', 'vae_decoder'].flatMap(name => [`load:${name}`, `run:${name}`, `release:${name}`])).flat());
});

test('failed drawing releases its network before a retry', async () => {
  let released = 0;
  const sessions = new DiffusionSessions(true, async () => ({ async release() { released++; } }));
  await expect(sessions.use('unet', async () => { throw new Error('GPU allocation failed'); })).rejects.toThrow('GPU allocation failed');
  expect(released).toBe(1);
  await sessions.use('unet', async () => 'retry');
  expect(released).toBe(2);
});

test('standard drawing retains networks and cleans up a partial load failure', async () => {
  let loads = 0, released = 0;
  const sessions = new DiffusionSessions(false, async name => {
    loads++;
    if (name === 'bad') throw new Error('load failed');
    return { async release() { released++; } };
  });
  await sessions.preload(['text_encoder']);
  await sessions.use('text_encoder', async () => 1);
  await sessions.use('text_encoder', async () => 2);
  expect(loads).toBe(1); expect(released).toBe(0);
  await expect(sessions.preload(['bad'])).rejects.toThrow('load failed');
  expect(released).toBe(1);
});
