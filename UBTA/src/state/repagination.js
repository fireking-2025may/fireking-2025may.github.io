/** Coordinates the expensive, asynchronous replacement of a paginated preview. */
export function createRepaginationScheduler({ render, capture = () => undefined, restore = () => {}, delay = 500, setTimer = setTimeout, clearTimer = clearTimeout }) {
  let timer = null;
  let generation = 0;
  async function run(mine) {
    if (mine !== generation) return false;
    timer = null;
    const view = capture();
    const isCurrent = () => mine === generation;
    const result = await render({ generation: mine, isCurrent, view });
    if (!isCurrent()) return false;
    await restore(view, result);
    return true;
  }
  function request({ immediate = false } = {}) {
    const mine = ++generation;
    if (timer !== null) clearTimer(timer);
    timer = null;
    if (immediate) return run(mine);
    timer = setTimer(() => run(mine), delay);
    return mine;
  }
  function cancel() {
    generation += 1;
    if (timer !== null) clearTimer(timer);
    timer = null;
  }
  return { request, cancel, get generation() { return generation; } };
}
