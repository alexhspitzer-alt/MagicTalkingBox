# Magic Talking Box

A small browser AI workstation: **WebGPU inference, one local model, offline app
and model caches, and a conversation saved on your device.** No backend, API key,
account, inference service, or CPU fallback.

## First milestone

1. Open the hosted app using HTTPS in a browser with WebGPU support.
2. Click **Load model** while online. The first download includes ~194 MiB of
   weights plus the tokenizer and GPU runtime. Loading may take several minutes.
3. Wait for **Offline ready**, with both app and model caches complete.
4. Disconnect and chat. Reload the page offline, click **Load model** again, and
   continue: the GPU must load the cached weights into memory, but no download is
   needed. The conversation survives reloads.

Browser storage is scoped to the origin and browser profile. The app requests
persistent storage from the load-button gesture and reports whether the browser
grants it. When persistence is denied, the cache is best effort and can be evicted
under storage pressure. Clearing site data or using another browser/profile needs
a new download. Private browsing may block or discard storage. Keep the same URL.

## Run locally

Requires Node.js 22.12+ (Node 24 recommended).

```sh
npm ci
npm run dev
```

Development mode deliberately disables the app service worker. To test the
offline milestone, use the production build:

```sh
npm run build
npm run preview
```

Open the localhost URL printed by Vite. A phone accessing a computer's ordinary
HTTP LAN address is **not** a secure context: use HTTPS hosting for phone tests.
The relative asset paths also work under a GitHub Pages repository subpath.
To enable hosting, choose **Settings → Pages → Source → GitHub Actions** once.
The included workflow builds, runs smoke tests, and deploys `main` to GitHub
Pages. After enabling hosting, rerun the workflow from the Actions tab if needed.

## Model and runtime

- WebLLM **0.2.85**, pinned with a lockfile.
- **SmolLM2-360M-Instruct-q4f32_1-MLC**: open-weight instruct model, 360M parameters,
  4-bit quantized weights with float32 computation. This prebuilt WebGPU model
  needs no `shader-f16` feature. It is a constrained-device baseline, with limited
  reasoning and factual accuracy.
- Context: **4,096 tokens** including conversation and output.
- Output: up to **1,024 tokens per response**, with **no wall-clock timeout**.
  Stop is an explicit user action. Slow devices can finish at their own pace.
- WebLLM's published GPU memory estimate is **~580 MB**; browser overhead and
  temporary allocations also consume RAM. Actual GPU allocation is not exposed.
- Generated text streams from a dedicated web worker. Decode speed comes from
  WebLLM's actual token statistics at completion, not character counts.
- The page reports JS heap use where supported, explicitly labeled as page heap,
  plus total origin storage use. These are not measurements of VRAM or total RAM.

WebGPU support varies by OS, browser, GPU, and driver. API availability alone
does not guarantee the model fits. Startup requests a usable adapter/device;
loading reports runtime GPU-limit and allocation errors. Context overflow is
reported rather than silently deleting saved history; start a new chat when needed.
Mobile operating systems can suspend hidden tabs or kill them under memory
pressure. There is no promise of background execution; keep the tab visible.

## Modules

| Module | Responsibility |
| --- | --- |
| `src/device.ts` | Secure-context/WebGPU detection and available memory telemetry |
| `src/model.ts` | One supported model and its context configuration |
| `src/inference.ts` | Model loading, worker lifecycle, streaming, interruption |
| `src/worker.ts` | WebLLM worker handler; actual GPU inference |
| `src/persistence.ts` | Chat storage, persistence request, complete artifact-cache checks |
| `src/ui.ts` | Text rendering and controls |
| `src/main.ts` | App lifecycle and event orchestration |
| `scripts/build.mjs` | Build and versioned precache of every app/worker asset |

The app service worker only caches the app's own files. WebLLM owns model weights,
tokenizer, config and WASM caches. Offline readiness verifies all manifest shards
and the selected tokenizer/config/runtime, not just a remembered success flag.
The initial model and runtime downloads contact Hugging Face and MLC's GitHub
distribution. Chat messages are never uploaded by this app. Text is rendered as
plain text, so model responses do not execute HTML.

## Verification

```sh
npm run build
npx playwright install chromium
npm test
```

Smoke tests cover unsupported devices, narrow-screen layout, safely rendered chat
history, an offline app reload at `/MagicTalkingBox/`, the cached inference worker,
and detection of an incomplete model cache. They do not simulate successful AI
inference. The opt-in real-model test downloads the actual model, generates offline,
reloads it offline, generates again, and checks for zero remote requests:

```sh
REAL_MODEL=1 npm test -- --grep 'real model'
```

Use a WebGPU-capable test host. `CHROMIUM_PATH` can select an existing compatible
Chromium binary. Test time limits are harness limits, not app generation limits.
`REAL_CACHE=1 npm test -- --grep 'real cache'` checks actual model downloads and an
offline page/model reload without waiting for generation.

Initial verification: the production build and three smoke tests passed. The real
model downloaded and initialized on Chromium 153's software WebGPU adapter, then
reloaded successfully after an offline page refresh with zero remote requests.
Its offline reply did not finish within the harness's five-minute limit. Full
offline chat needs confirmation on a device with a supported GPU; the app itself
has no generation time limit.

Model source: https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q4f32_1-MLC

Runtime documentation: https://webllm.mlc.ai/docs/user/advanced_usage.html
