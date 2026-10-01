export async function detectDevice(): Promise<string> {
  if (!isSecureContext) throw new Error('WebGPU needs a secure page. Open this app over HTTPS or localhost.');
  if (!navigator.gpu) throw new Error('This browser/device does not support WebGPU. Try an up-to-date browser with WebGPU enabled on a supported device.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU is present, but no usable GPU adapter is available. Check your browser and device GPU support.');
  // Validate that a device can actually be created before downloading any weights.
  const device = await adapter.requestDevice();
  device.destroy();
  const info = adapter.info;
  if (!info) return 'WebGPU adapter (browser hides device name)';
  return [info.description, info.vendor, info.architecture, info.device].filter(Boolean).join(' · ') || 'WebGPU adapter (browser hides device name)';
}

export function memoryInfo(): string {
  const memory = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
  return memory ? `Page JS heap: ${(memory.usedJSHeapSize / 1048576).toFixed(0)} MiB · GPU use unavailable` : 'Actual GPU / RAM use unavailable';
}
