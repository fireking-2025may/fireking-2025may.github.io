export class PrintLifecycle {
  constructor({
    flush,
    cancelPagination,
    render,
    prepare,
    clearPresentation,
    print,
    restore,
    onAfterPrint,
    onError = () => {},
    stabilize = () => true,
    setFallback = (callback) => setTimeout(callback, 1000),
    clearFallback = clearTimeout,
  }) {
    Object.assign(this, {
      flush,
      cancelPagination,
      render,
      prepare,
      clearPresentation,
      print,
      restore,
      onAfterPrint,
      onError,
      stabilize,
      setFallback,
      clearFallback,
    });
    this.busy = false;
  }

  async run() {
    if (this.busy) return false;
    this.busy = true;
    let context;
    let prepared = false;
    let fallback;
    let removeAfterPrint = () => {};
    const cleanup = () => {
      if (!this.busy) return;
      this.busy = false;
      if (fallback !== undefined) this.clearFallback(fallback);
      removeAfterPrint();
      if (prepared) this.restore(context);
    };
    try {
      context = this.prepare();
      prepared = true;
      this.flush();
      this.cancelPagination();
      if (!(await this.render()))
        throw new Error('The document could not be prepared for printing.');
      // The first pass calculates the new page map. The second pass lays out the
      // contents page with that map, so the printed TOC matches the final pages.
      if (!(await this.render()))
        throw new Error(
          'The document contents could not be reconciled for printing.',
        );
      if (!(await this.stabilize()))
        throw new Error('The paginated print layout did not become stable.');
      this.clearPresentation();
      removeAfterPrint = this.onAfterPrint(cleanup);
      this.print();
      if (this.busy) fallback = this.setFallback(cleanup);
      return true;
    } catch (error) {
      cleanup();
      this.onError(error);
      return false;
    }
  }
}
