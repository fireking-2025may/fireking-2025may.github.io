import { safeHref } from '../state/schema.js';

export function followEditorLink(
  href,
  { navigateInternal, openExternal, openMailto, report = () => {} },
) {
  const safe = safeHref(href);
  if (!safe) {
    report('This link is not safe to open');
    return false;
  }
  if (safe.startsWith('#')) {
    const opened = navigateInternal(safe.slice(1));
    if (!opened) report('The linked document destination no longer exists');
    return opened;
  }
  if (/^https?:/i.test(safe)) {
    openExternal(safe);
    return true;
  }
  if (/^mailto:/i.test(safe)) {
    openMailto(safe);
    return true;
  }
  return false;
}

export function handleEditableLinkClick(event, openLink) {
  const link = event.target?.closest?.('.editable-runs a');
  if (!link || event.button !== 0) return false;
  event.preventDefault();
  openLink(link.getAttribute('href'));
  return true;
}

export function createEditableLinkClickHandler({ handleClick, openLink }) {
  return function editableLinkClick(event) {
    return handleClick(event, openLink);
  };
}
