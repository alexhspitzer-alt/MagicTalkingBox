# Magic Talking Box

A browser-local AI workstation with **text and drawing modes**, WebGPU inference,
persistent model caches, saved conversations and a downloadable PNG gallery.
No server-side inference, API key or account is required.

## Use it

1. Open https://alexhspitzer-alt.github.io/MagicTalkingBox/ on a WebGPU device.
2. Choose **Text / chat** or **Image / drawing**, then a model.
3. Click **Load model** while online. Only the selected model downloads.
4. Wait for **Offline ready**, with both app and selected-model caches complete.
5. Disconnect and generate. After an offline reload, click **Load model** to load
   cached weights back into GPU memory. Conversations and images survive reloads.

Switching modes/models terminates the previous worker and releases its GPU
resources. Its downloaded weights remain cached. Returning to a cached model
requires loading it again, but not downloading it again. Clear gallery removes
saved images; New chat removes the saved conversation. Neither deletes weights.

## Model choices

| Mode | Model | Approximate weights | Context / image |
| --- | --- | --- | --- |
| Text | SmolLM2 360M Instruct (default) | 194 MiB | 4,096 tokens |
| Text | Qwen2.5 0.5B Instruct | 265 MiB | 4,096 tokens |
| Text | Llama 3.2 1B Instruct | 663 MiB | 4,096 tokens |
| Text | Qwen2.5 1.5B Instruct | 828 MiB | 4,096 tokens |
| Text | Qwen2.5 3B Instruct | 1,656 MiB | 4,096 tokens |
| Text | Llama 3.2 3B Instruct | 1,724 MiB | 4,096 tokens |
| Text | Llama 3.2 3B, lower memory (q4f16) | 1,724 MiB | 2,048 tokens |
| Text | Qwen2.5 7B Instruct | 4,086 MiB | 4,096 tokens |
| Image | Tiny SD LCM, smaller memory | ~947 MiB | 256 × 256; eight guided LCM steps |
| Image | Stable Diffusion Turbo (default) | 2.34 GiB | 512 × 512; one diffusion step |
| Image | Stable Diffusion Turbo, low memory | Same shared 2.34 GiB | 256 × 256; one network at a time |
| Image | Janus Pro 1B | 2.13 GiB | 384 × 384; 576 image tokens |
| Image | Janus 1.3B | 2.13 GiB | 384 × 384; 576 image tokens |

Download sizes exclude tokenizers, configs and runtimes. Images need several GB
of available device memory **in addition to** disk storage; they are substantially
heavier than the small text baseline. WebGPU support alone does not guarantee a
model fits. Loading/allocation errors are shown explicitly.

Text uses WebLLM 0.2.85. Standard q4f32 choices need no shader-f16 feature.
The lower-memory Llama 3B variant uses q4f16 and requires shader-f16, checked
before downloading weights. It has a 2,048-token context and a 512-token output
limit. The published GPU estimate at the reference 4,096-token context is
2,264 MB, compared with 2,952 MB for its q4f32 counterpart; the shorter context
reduces KV-cache allocation further. Actual phone compatibility remains unverified.
Its weights are cached separately; existing caches are retained.

A tab crash after download can reflect peak RAM/GPU use during initialization.
The origin storage reading measures disk usage, not live inference memory.
A cache quota failure normally reports QuotaExceededError.
The standard larger text models retain the 4,096-token context. Published GPU allocation
estimates are about 2,894 MB for Qwen 3B, 2,952 MB for Llama 3B and 5,900 MB for
Qwen 7B, plus browser overhead. Their download sizes are measured from the
published weight manifests. Try a 3B model before 7B on phones; cached weights
fitting on disk do not guarantee that a loaded model fits in device memory.
Responses stream from a dedicated worker, with a 1,024 output token limit and no
wall-clock timeout. Speed is WebLLM's actual decode-token statistic.

Drawing uses pinned Transformers.js 3.8.1 and ONNX Runtime Web
1.22.0-dev.20250409-89f8206ba4. SD-Turbo uses the one-step pipeline from Microsoft's
WebGPU example and requires shader-f16. Janus uses mixed q4 / float32 weights:
its input embedding graph uses WASM for upstream compatibility, while its language
and image decoders run on WebGPU. Janus progress reports image tokens/s, which
cannot be predicted from another text model's speed. Completed image speed is
reported as images/min and seconds/image. There are no generation timeouts.
Stopping a drawing terminates the worker; load the cached model again to continue.

**SD-Turbo · low memory** streams the same weights into the existing disk cache
without initializing all three networks. Each drawing loads the prompt encoder,
UNet, then VAE decoder in sequence, awaiting session release between stages.
Small CPU tensors carry results between networks; each stage still runs on WebGPU.
This reduces simultaneous resident weights and trades speed for memory. Low-memory
mode uses 256 × 256 output (32 × 32 latents) to reduce activation buffers; standard
mode keeps 512 × 512. Both use one diffusion step. Session creation uses a temporary
Blob URL so ORT can discard its source buffer after copying into WASM, rather than
retaining an application-owned model buffer throughout graph initialization.
The temporary URL is revoked on success and failure. This still requires a
model-sized WASM allocation and GPU weights; it does not stream weights into GPU memory. Standard SD-Turbo and low-memory
mode reuse the exact same weight URLs, so switching does not duplicate downloads.
The largest network and its initialization buffers must still fit; this is not a
guarantee for every phone. The first real network allocation occurs when drawing.
Recreating sessions from cached files makes every image slower, including offline.

**Tiny SD LCM · smaller memory** is a separate, genuinely smaller model based on
Segmind Tiny SD and an LCM adapter. The pinned ONNX export is
`akameswa/lcm-tiny-sd-onnx-fp16` at `3b423c4e1d61d50c56efdde24592862be68c8574`.
Its drawing weights are 646,809,600 bytes (~617 MiB), approximately 63% smaller
than SD-Turbo's UNet. The text-to-image files total ~947 MiB; VAE encoder files
are not downloaded. This is a separate cache; existing SD-Turbo downloads remain.
The pipeline draws at 256 × 256 to limit activation memory on phones, using an experimental eight-step/guidance-2
profile, the export's scaled-linear/epsilon LCM schedule, and SD 1.5 latent scaling.
The encoder, UNet (all eight steps), and decoder run in sequence with CPU stage
outputs; each network is released before the next starts. External ONNX data
files are cached and passed explicitly to ORT, including during offline loading.
Float16 GPU support is required. Smaller weights improve the prospect of fitting
on a phone, but do not guarantee successful initialization or good image quality.
Graph inputs, dynamic dimensions, external data paths, and runtime operator
support were inspected without downloading complete weights or generating images.

The app displays GPU/device, selected model and size, context or output size,
generation speed, elapsed time, origin storage, and page JS heap where exposed.
Actual VRAM/total system RAM usage is not available through standard browser APIs.

## Persistence and offline support

The production service worker precaches all bundled app/worker assets and local
ONNX WASM/glue files. It never deletes model caches when updating the app.
WebLLM owns its model/config/WASM caches. Completeness checks verify **every weight
shard**, tokenizer, configuration and GPU runtime for the selected text model.
Image downloads use a dedicated Cache API store; after successful loading, a
manifest records all used weights, tokenizer and configuration URLs. Offline
readiness rechecks every entry. Model files are never considered cached merely
because a past download succeeded. Tokenizers are loaded during setup, so the
first offline drawing does not need a lazy tokenizer download.

Chat uses localStorage; image PNG blobs and their prompts/model/timing metadata
use IndexedDB. Prompts and generated outputs are never uploaded. Text is rendered
as plain text. The PNG gallery offers local downloads, including if saving fails.

Storage belongs to the same origin and browser profile. The app requests durable
storage from the load gesture and shows whether the browser grants it. Best-effort
storage may be evicted under pressure; clearing site data requires new downloads.
Keep the tab visible: mobile operating systems may suspend/kill background tabs.

## Development and deployment

Requires Node.js 22.12+ (24 recommended).

```sh
ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm ci
npm run dev
```

The environment variable skips an unused Node CUDA binary download; inference
uses browser WebGPU. Development mode disables the app service worker. For offline
checks use the production build:

```sh
npm run build
npm run preview
```

HTTPS or localhost is required for WebGPU. Relative assets work at the GitHub
Pages repository subpath. The included workflow builds, runs lightweight smoke
tests, and deploys `main` to GitHub Pages using GitHub Actions.

## Modules

| Module | Responsibility |
| --- | --- |
| `src/model.ts` | Shared text/image model catalog |
| `src/text-model.ts` | Pinned WebLLM model configurations |
| `src/device.ts` | WebGPU detection and available memory telemetry |
| `src/inference.ts`, `src/worker.ts` | Text worker lifecycle, loading, streaming and stop |
| `src/image-inference.ts`, `src/image-worker.ts` | Image worker protocol, loading and generation |
| `src/diffusion.ts`, `src/diffusion-sessions.ts` | SD-Turbo pipeline and retained/staged network lifecycle |
| `src/persistence.ts` | Chat, storage persistence, text cache verification and app caching |
| `src/image-cache.ts` | Image model asset storage and complete manifests |
| `src/image-history.ts` | IndexedDB PNG/prompt history |
| `src/ui.ts` | Safe chat/gallery rendering and controls |
| `src/main.ts` | Mode/model selection and app lifecycle |
| `scripts/build.mjs` | Build and complete offline app precache |

## Verification

```sh
npm run build
npx playwright install chromium
npm test
```

Default smoke tests cover unsupported devices, narrow layouts, mode/model menus
without downloads, text and image history restored offline, PNG download links,
worker release on switching, incomplete caches, cached runtime files, and actual
image-worker startup offline. Image UI integration uses a tiny fixture through
the worker protocol; it does **not** run expensive inference. No live image model
download/generation is performed by default. Image output quality and device GPU
compatibility still require use on the target device.
Small mocked-session checks verify staged release order, repeated drawings, and
cleanup on errors, without loading model weights or running GPU inference.

Opt-in existing text checks: `REAL_MODEL=1 npm test -- --grep 'real model'` downloads
and generates; `REAL_CACHE=1 npm test -- --grep 'real cache'` checks a real offline
model reload without generation. These harness time limits do not apply to the app.
`CHROMIUM_PATH` selects a compatible existing Chromium binary.

## Model/runtime sources

- [WebLLM](https://webllm.mlc.ai/docs/)
- [Microsoft SD-Turbo WebGPU example](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/sd-turbo)
- [SD-Turbo ONNX files](https://huggingface.co/schmuell/sd-turbo-ort-web)
- [Janus Pro 1B ONNX and reference usage](https://huggingface.co/onnx-community/Janus-Pro-1B-ONNX)
- [Janus 1.3B ONNX and reference usage](https://huggingface.co/onnx-community/Janus-1.3B-ONNX)

Each model retains its publisher's license; follow the linked model cards.

Tiny SD image history includes separate network-loading and computation timings
for the prompt encoder, all eight guided drawing steps, and decoder. Timings measure
wall-clock seconds, not GPU utilization or memory usage. The 512 × 512 trial caused browser crashes on the target phone during the first
drawing pass, so output returns to 256 × 256. Weight downloads and cache URLs
are unchanged, and stage timings remain available.

Online page navigation checks the server for a fresh app shell, falling back to
the installed page offline. App installation bypasses stale HTTP cache entries;
service-worker script updates also bypass HTTP caching. Model caches are separate
and remain intact when the app updates.

The original four-step/guidance-1 profile completed on the phone but produced
poor results. The guided profile uses conditional and unconditional predictions
in separate batch-one calls, then combines them with CFG scale 2. It increases
computation without doubling the UNet batch. Non-finite predictions, latents,
and decoded pixels now raise errors instead of silently rendering corrupted
images. Static review found matching tokenizer/padding, dynamic spatial shapes,
and sampled ONNX weights consistent with the source at float16 precision.
Full output quality and GPU numerical correctness remain unverified.
