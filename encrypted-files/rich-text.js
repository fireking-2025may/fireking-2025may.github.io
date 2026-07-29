const sameFormatting = (left, right) =>
  left.highlight === right.highlight && left.link?.href === right.link?.href;

const formattingAt = (runs, characterIndex) => {
  let offset = 0;
  for (const run of runs) {
    const end = offset + run.text.length;
    if (characterIndex < end) return run;
    offset = end;
  }
  return runs.at(-1) || {};
};

const appendCharacter = (output, character, formatting) => {
  const run = {
    text: character,
    ...(formatting.highlight != null
      ? { highlight: formatting.highlight }
      : {}),
    ...(formatting.link != null
      ? { link: { href: formatting.link.href } }
      : {}),
  };
  const previous = output.at(-1);
  if (previous && sameFormatting(previous, run)) previous.text += character;
  else output.push(run);
};

/** Apply a plain-text edit while retaining formatting on unchanged text. */
export function applyPlainTextEdit(runs, value) {
  const nextText = value.replace(/\r\n?/g, "\n");
  const previousText = runs.map((run) => run.text).join("");
  let prefixLength = 0;
  while (
    prefixLength < previousText.length &&
    prefixLength < nextText.length &&
    previousText[prefixLength] === nextText[prefixLength]
  ) {
    prefixLength += 1;
  }

  let suffixLength = 0;
  while (
    suffixLength < previousText.length - prefixLength &&
    suffixLength < nextText.length - prefixLength &&
    previousText.at(-1 - suffixLength) === nextText.at(-1 - suffixLength)
  ) {
    suffixLength += 1;
  }

  const output = [];
  for (let index = 0; index < nextText.length; index += 1) {
    let sourceIndex = prefixLength;
    if (index < prefixLength) sourceIndex = index;
    else if (index >= nextText.length - suffixLength) {
      sourceIndex = previousText.length - (nextText.length - index);
    } else if (sourceIndex === previousText.length) {
      sourceIndex = Math.max(0, sourceIndex - 1);
    }
    appendCharacter(output, nextText[index], formattingAt(runs, sourceIndex));
  }

  if (!output.length) output.push({ text: "" });
  return output;
}
