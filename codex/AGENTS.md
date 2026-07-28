Before starting, apply these rules consistently:

Prefer small named functions over compressed one-line expressions.

Give each module one clear responsibility.

Keep browser/DOM operations in editor/ and document rules in state/.

Pass dependencies as arguments instead of reading hidden global variables.

Prefer straightforward if statements and early returns over clever abstractions.

Keep modules reasonably sized—roughly 100–250 readable lines, depending on the responsibility.

Use names such as addTableRow, selectedBlock, and renderDocumentPage, rather than generic names such as x, b, c, or manage.

Add tests as each function is extracted, not at the end.

Avoid introducing a framework or a complex state-management library; plain JavaScript modules are sufficient.

The current code especially needs normal formatting: initialization, rendering, mutation, event binding, persistence, and navigation are compressed into dense lines in main.js
