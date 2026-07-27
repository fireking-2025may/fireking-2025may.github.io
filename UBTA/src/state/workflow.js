export const WORKFLOW_STATUSES = ['Draft', 'In review', 'Approved', 'Archived'];

export function normaliseStatus(value) {
  return WORKFLOW_STATUSES.includes(value) ? value : 'Draft';
}

export function workflowTransition(from, to) {
  const current = normaliseStatus(from), next = normaliseStatus(to);
  if (current === next) return { allowed: true, confirm: false, snapshot: false };
  return {
    allowed: true,
    confirm: ['Approved', 'Archived'].includes(next) || ['Approved', 'Archived'].includes(current),
    snapshot: ['Approved', 'Archived'].includes(next),
  };
}
