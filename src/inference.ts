import { CreateWebWorkerMLCEngine, type MLCEngineInterface, type InitProgressReport, type CompletionUsage } from '@mlc-ai/web-llm';
import { APP_CONFIG, MODEL_ID, MAX_OUTPUT_TOKENS } from './model';
import type { Message } from './persistence';

export class Inference {
  private engine?: MLCEngineInterface;
  private worker?: Worker;
  private rejectFailure?: (error: Error) => void;
  private failure?: Error;
  constructor(private onFatal: (error: Error) => void) {}

  private async withWorkerFailure<T>(operation: Promise<T>): Promise<T> {
    if (this.failure) throw this.failure;
    const failed = new Promise<never>((_, reject) => { this.rejectFailure = reject; });
    try { return await Promise.race([operation, failed]); }
    finally { this.rejectFailure = undefined; }
  }

  async load(onProgress: (report: InitProgressReport) => void) {
    this.worker?.terminate();
    this.engine = undefined;
    this.failure = undefined;
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    const fail = () => {
      this.failure = new Error('The inference worker stopped unexpectedly. Your device may have run out of memory. Reload the model to retry.');
      this.rejectFailure?.(this.failure);
      this.onFatal(this.failure);
    };
    this.worker.addEventListener('error', fail);
    this.worker.addEventListener('messageerror', fail);
    this.engine = await this.withWorkerFailure(CreateWebWorkerMLCEngine(this.worker, MODEL_ID, {
      appConfig: APP_CONFIG, initProgressCallback: onProgress,
    }));
  }

  async generate(messages: Message[], onText: (text: string) => void, onUsage: (usage: CompletionUsage) => void): Promise<string | null> {
    if (!this.engine) throw new Error('Load the model before sending a message.');
    return this.withWorkerFailure((async () => {
      const stream = await this.engine!.chat.completions.create({
        messages: [{ role: 'system', content: 'You are a helpful assistant. Answer clearly and honestly.' }, ...messages],
        stream: true, stream_options: { include_usage: true },
        max_tokens: MAX_OUTPUT_TOKENS, temperature: 0.7,
      });
      let finishReason: string | null = null;
      for await (const chunk of stream) {
        const choice = chunk.choices[0];
        if (choice?.delta.content) onText(choice.delta.content);
        if (choice?.finish_reason) finishReason = choice.finish_reason;
        if (chunk.usage) onUsage(chunk.usage);
      }
      return finishReason;
    })());
  }
  async stop() { await this.engine?.interruptGenerate(); }
}
