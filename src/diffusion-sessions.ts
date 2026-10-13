type Releasable = { release(): Promise<void> };

// Staged mode awaits resource release before the next network is created.
// Results passed between stages must be CPU tensors, independent of a session.
export class DiffusionSessions<S extends Releasable> {
  private retained = new Map<string, S>();
  constructor(private staged: boolean, private create: (name: string) => Promise<S>) {}

  async preload(names: string[]) {
    if (this.staged) return;
    try {
      for (const name of names) this.retained.set(name, await this.create(name));
    } catch (error) {
      await this.release();
      throw error;
    }
  }

  async use<T>(name: string, operation: (session: S) => Promise<T>): Promise<T> {
    if (!this.staged) {
      const session = this.retained.get(name);
      if (!session) throw new Error(`Image network is not loaded: ${name}`);
      return operation(session);
    }
    const session = await this.create(name);
    try { return await operation(session); }
    finally { await session.release(); }
  }

  async release() {
    const sessions = [...this.retained.values()];
    this.retained.clear();
    await Promise.all(sessions.map(session => session.release()));
  }
}
