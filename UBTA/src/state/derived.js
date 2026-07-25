function blockText(block) {
  const runs = block.runs || block.items?.flatMap(item => item.runs) || [];
  return runs.map(run => run.text).join(' ').replace(/\s+/g, ' ').trim();
}

function isInReview(step) {
  return step.blocks.some(block =>
    (block.runs || block.items?.flatMap(item => item.runs) || []).some(run => run.highlight)
  );
}

export function generatedProposal(title, summary) {
  const subject = title.trim() || 'Complete the proposed transaction step';
  const detail = summary.trim();
  return detail ? `Proposal: ${subject} — ${detail}` : `Proposal: ${subject}.`;
}

/** Return render-only step content without mutating or committing the document. */
export function deriveStepDescriptors(document) {
  return document.steps.map((step, index) => {
    const stepNumber = index + 1;
    const summary = step.summary || step.blocks.map(blockText).find(Boolean) || '';
    const generatedText = generatedProposal(step.title, summary);
    const overridden = Boolean(step.proposal?.trim());
    return {
      stepNumber, id: step.id,
      displayTitle: `Step ${stepNumber}. ${step.title}`,
      anchor: `anchor-${step.id}`,
      proposalAnchor: `anchor-proposal-${step.id}`,
      proposalText: overridden ? step.proposal : generatedText,
      generatedText,
      proposalSource: overridden ? 'override' : 'generated',
      reviewState: isInReview(step) ? 'review' : 'clear',
      tocLabel: `Step ${stepNumber}. ${step.title}`
    };
  });
}
