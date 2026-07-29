import { GenerationGate } from './interactions.js';
export class RepaginationScheduler {
  constructor(
    renderer,
    { delay = 500, capture = () => null, restore = () => {} } = {},
  ) {
    this.renderer = renderer;
    this.delay = delay;
    this.capture = capture;
    this.restore = restore;
    this.gate = new GenerationGate();
    this.timer = null;
  }
  request({ immediate = false } = {}) {
    this.cancel();
    const generation = this.gate.next(),
      view = this.capture();
    if (immediate) return this.run(generation, view);
    return new Promise((resolve) => {
      this.queuedResolve = resolve;
      this.timer = setTimeout(() => {
        this.timer = null;
        this.queuedResolve = null;
        this.run(generation, view).then(resolve);
      }, this.delay);
    });
  }
  cancel() {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
      this.queuedResolve?.(false);
      this.queuedResolve = null;
    }
    this.gate.cancel();
  }
  async run(generation, view) {
    const isCurrent = () => this.gate.isCurrent(generation);
    const completed = await this.renderer({ isCurrent });
    if (!completed || !isCurrent()) return false;
    await this.restore(view);
    return isCurrent();
  }
}
