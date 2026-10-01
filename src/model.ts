import { prebuiltAppConfig, type AppConfig } from '@mlc-ai/web-llm';

export const MODEL_ID = 'SmolLM2-360M-Instruct-q4f32_1-MLC';
export const MODEL_NAME = 'SmolLM2 360M Instruct · q4f32';
export const CONTEXT_SIZE = 4096;
export const MAX_OUTPUT_TOKENS = 1024;
const record = prebuiltAppConfig.model_list.find(model => model.model_id === MODEL_ID);
if (!record) throw new Error(`The pinned WebLLM release does not contain ${MODEL_ID}.`);
export const MODEL = { ...record, overrides: { ...record.overrides, context_window_size: CONTEXT_SIZE } };
export const APP_CONFIG: AppConfig = { model_list: [MODEL], cacheBackend: 'cache' };
export const MODEL_BASE = `${MODEL.model.replace(/\/$/, '')}/resolve/main/`;
