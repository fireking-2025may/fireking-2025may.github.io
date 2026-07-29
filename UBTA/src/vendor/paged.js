/*! Offline pagination engine for the UBTA editor.
 * The public surface deliberately matches the part of Paged.js used by the app.
 * Content is measured in a connected, hidden staging area and is fragmented into
 * editable page-sized DOM trees. */
(function () {
  const PAGE = 'pagedjs_page';

  function pageFor(sheet, target) {
    const page = document.createElement('div');
    page.className = PAGE;
    page.tabIndex = -1;
    const box = document.createElement('div');
    box.className = 'pagedjs_pagebox';
    box.append(sheet);
    page.append(box);
    target.append(page);
    return page;
  }

  function overflows(sheet) {
    const body = sheet.querySelector('.document-body');
    if (!body) return false;
    const style = getComputedStyle(sheet);
    const limit =
      sheet.getBoundingClientRect().bottom - parseFloat(style.paddingBottom);
    return body.getBoundingClientRect().bottom > limit + 0.5;
  }

  function cloneShell(source, continuation) {
    const sheet = document.createElement(source.tagName);
    sheet.className = source.className;
    sheet.tabIndex = -1;
    for (const [key, value] of Object.entries(source.dataset))
      sheet.dataset[key] = value;
    if (!continuation && source.id) sheet.id = source.id;
    else if (source.id) sheet.dataset.anchorId = source.id;
    const header = source.querySelector(':scope > .page-header');
    if (header) sheet.append(header.cloneNode(true));
    const heading = source.querySelector(
      ':scope > .step-heading, :scope > .section-title',
    );
    if (heading) {
      if (!continuation) sheet.append(heading.cloneNode(true));
      else {
        const continued = document.createElement('div');
        continued.className = 'continuation-heading';
        const title =
          heading.querySelector('.step-title')?.value ||
          heading.textContent.trim();
        continued.textContent = `${title} — continued`;
        sheet.append(continued);
      }
    }
    const body = document.createElement('div');
    body.className = 'document-body';
    sheet.append(body);
    return sheet;
  }

  function textLength(root) {
    return [...root.querySelectorAll('[contenteditable=true]')].reduce(
      (n, el) => n + el.textContent.length,
      0,
    );
  }

  function textBoundary(root, offset) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let remaining = offset;
    let node;
    while ((node = walker.nextNode())) {
      if (remaining <= node.data.length) return [node, remaining];
      remaining -= node.data.length;
    }
    return [root, root.childNodes.length];
  }

  function richTextSlice(source, start, end) {
    const range = document.createRange();
    range.setStart(...textBoundary(source, start));
    range.setEnd(...textBoundary(source, end));
    return range.cloneContents();
  }

  function splitTextNode(block, fit) {
    const editable = block.querySelector('[contenteditable=true]');
    if (!editable || editable.querySelector('table,img')) return null;
    const original = editable.cloneNode(true);
    const text = original.textContent;
    if (text.length < 2) return null;
    let low = 1,
      high = text.length - 1,
      best = 0;
    while (low <= high) {
      const middle = (low + high) >> 1;
      editable.replaceChildren(richTextSlice(original, 0, middle));
      if (fit()) {
        best = middle;
        low = middle + 1;
      } else high = middle - 1;
    }
    if (!best) {
      editable.replaceChildren(...original.cloneNode(true).childNodes);
      return null;
    }
    const breakAt = text.lastIndexOf(' ', best);
    best = breakAt > best / 2 ? breakAt + 1 : best;
    editable.replaceChildren(richTextSlice(original, 0, best));
    const rest = block.cloneNode(true);
    rest
      .querySelector('[contenteditable=true]')
      .replaceChildren(richTextSlice(original, best, text.length));
    block.dataset.fragmentPart = 'true';
    rest.dataset.fragmentPart = 'true';
    return rest;
  }

  function splitList(block, fit) {
    const list = block.querySelector(':scope > ol, :scope > ul');
    if (!list || list.children.length < 2) return null;
    const rest = block.cloneNode(true);
    const items = [...list.children];
    let keep = items.length;
    while (keep > 1 && !fit()) items[--keep].remove();
    if (!fit()) {
      list.replaceChildren(
        ...rest.querySelector(':scope > ol, :scope > ul').cloneNode(true)
          .children,
      );
      return null;
    }
    const restList = rest.querySelector(':scope > ol, :scope > ul');
    [...restList.children].slice(0, keep).forEach((el) => el.remove());
    block.querySelector(':scope > .list-trailing-hit-area')?.remove();
    block.dataset.fragmentPart = rest.dataset.fragmentPart = 'true';
    return rest;
  }

  function splitTable(block, fit) {
    const rows = [...block.querySelectorAll('tbody > tr')];
    if (rows.length < 2) return null;
    const rest = block.cloneNode(true);
    let keep = rows.length;
    while (keep > 1 && !fit()) rows[--keep].remove();
    if (!fit()) {
      const sourceBody = rest.querySelector('tbody');
      block
        .querySelector('tbody')
        .replaceChildren(...sourceBody.cloneNode(true).children);
      return null;
    }
    [...rest.querySelectorAll('tbody > tr')]
      .slice(0, keep)
      .forEach((el) => el.remove());
    block.dataset.fragmentPart = rest.dataset.fragmentPart = 'true';
    return rest;
  }

  function fragmentBlock(block, sheet) {
    const fit = () => !overflows(sheet);
    if (block.classList.contains('list-block')) return splitList(block, fit);
    if (block.classList.contains('table-block')) return splitTable(block, fit);
    return splitTextNode(block, fit);
  }

  function fragmentGroup(source, target) {
    const blocks = [
      ...source.querySelectorAll(':scope > .document-body > .editable-block'),
    ].map((x) => x.cloneNode(true));
    let fragment = 0;
    let sheet = cloneShell(source, false);
    let page = pageFor(sheet, target);
    const decorate = () => {
      page.dataset.pageFragment = String(fragment);
      page.dataset.groupId = source.dataset.editableGroupId || '';
      page.dataset.groupType =
        source.dataset.groupType ||
        (source.classList.contains('section') ? 'section' : '');
      page.dataset.anchorId = source.id || '';
      sheet.dataset.fragmentIndex = String(fragment);
    };
    decorate();
    const nextPage = () => {
      fragment += 1;
      sheet = cloneShell(source, true);
      page = pageFor(sheet, target);
      decorate();
      return sheet.querySelector('.document-body');
    };
    let body = sheet.querySelector('.document-body');
    for (let index = 0; index < blocks.length; index += 1) {
      const block = blocks[index];
      body.append(block);
      if (!overflows(sheet)) continue;
      const hadPrevious = body.children.length > 1;
      if (hadPrevious) {
        const heading =
          block.matches('[data-type="paragraph"]') &&
          block.previousElementSibling?.matches('[data-type="heading"]')
            ? block.previousElementSibling
            : null;
        block.remove();
        heading?.remove();
        body = nextPage();
        if (heading) body.append(heading);
        body.append(block);
      }
      if (overflows(sheet)) {
        const rest = fragmentBlock(block, sheet);
        if (rest) blocks.splice(index + 1, 0, rest);
        else if (block.classList.contains('image-block'))
          block.classList.add('fit-page');
      }
    }
  }

  class Previewer {
    async preview(content, styles, target) {
      target.replaceChildren();
      const wasConnected = target.isConnected;
      if (!wasConnected) {
        target.classList.add('pagination-staging');
        document.body.append(target);
      }
      for (const source of [...content.children]) {
        if (source.querySelector(':scope > .document-body'))
          fragmentGroup(source, target);
        else pageFor(source.cloneNode(true), target);
      }
      target.classList.remove('pagination-staging');
      if (!wasConnected) target.remove();
      return { total: target.children.length, pages: [...target.children] };
    }
  }

  window.Paged = { Previewer };
})();
