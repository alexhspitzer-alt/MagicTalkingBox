import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', workers: 1, timeout: 30_000,
  use: {
    baseURL: 'http://127.0.0.1:4173/MagicTalkingBox/',
    proxy: process.env.TEST_PROXY_URL ? { server: process.env.TEST_PROXY_URL, bypass: '127.0.0.1,localhost' } : undefined,
    ignoreHTTPSErrors: Boolean(process.env.TEST_PROXY_URL),
    viewport: { width: 390, height: 844 },
    launchOptions: {
      executablePath: process.env.CHROMIUM_PATH,
      args: ['--no-sandbox', '--enable-unsafe-webgpu', '--use-angle=swiftshader'],
    },
  },
  webServer: { command: 'node tests/server.mjs', port: 4173, reuseExistingServer: !process.env.CI },
});
