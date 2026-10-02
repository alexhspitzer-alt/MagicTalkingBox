import { prebuiltAppConfig, type AppConfig } from '@mlc-ai/web-llm';
import { MODEL_ID, CONTEXT_SIZE } from './model';

export function textModel(id = MODEL_ID) {
  const record = prebuiltAppConfig.model_list.find(model => model.model_id === id);
  if (!record) throw new Error(`The pinned WebLLM release does not contain ${id}.`);
  return { ...record, overrides: { ...record.overrides, context_window_size: CONTEXT_SIZE } };
}
export function textConfig(id: string): AppConfig { return { model_list: [textModel(id)], cacheBackend: 'cache' }; }
export function modelBase(id = MODEL_ID) { return `${textModel(id).model.replace(/\/$/, '')}/resolve/main/`; }
