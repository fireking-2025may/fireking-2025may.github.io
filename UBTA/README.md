# UBTA Steps Plan Editor — phases 1–3 prototype

A client-side, desktop-first prototype for a constrained UBTA landscape steps plan. The serialisable schema is the source of truth; editor input is normalised to headings, paragraphs, three-level bullet/number lists, and text runs containing only review highlight and safe links.

## Development

Run `python3 -m http.server 8000 -d src`, then visit `http://localhost:8000`. No application server or network service is used by the editor. Run `npm test` for focused schema and sanitisation tests.

## Standalone build

Run `npm run build`, then open `dist/index.html` directly in current Chrome or Edge (`file:///…/dist/index.html`). It contains all CSS and JavaScript and makes no runtime network requests. State and undo history remain in memory and disappear when the tab closes.

## Implemented scope

* Versioned closed schema, normalisation, neutral seed document and fixed cover.
* Landscape A4 page preview with repeated report header, automatic linked contents, stable anchors and `Page X of Y` footers.
* In-preview constrained editing: body/heading styles, yellow review highlighting, approved links, lists and up to three levels, plain-text paste, session undo/redo, page navigation and debounced reflow.
* Stable-ID step management with editable titles, add/duplicate/delete/reorder controls, deletion safeguards, accessible position announcements, focus/scroll preservation, and undo/redo for every structural change.

## Deliberately deferred

Tables, editable/uploaded images and diagrams, generated proposals/acronyms, appendix entities, optional sections, collaboration/workflow/version history, persistence/authentication/APIs, exports/imports, and mobile/tablet layouts are not present. There are no placeholder controls for them.

## Typography and supplied assets

No licensed Neuzeit Grotesk webfont was present in the repository, so the isolated font token uses `"Century Gothic", "Futura", "Aptos", "Segoe UI", sans-serif`. Exact type fidelity depends on licensed Neuzeit files being supplied.

The named reference attachments, including `image(51).png`, were not present in the workspace and the PNG is intentionally excluded from this draft pull request. The prototype therefore uses a compact text fallback in the repeated header. Integrating and embedding the authoritative header remains blocked until that supplied asset is available and approved for inclusion.

## Prototype limitation

The local `src/vendor/paged.js` preserves the upstream MIT attribution and supplies the narrow `Paged.Previewer` interface needed by this fixed prototype. The upstream Paged.js distribution could not be retrieved in this environment. Replace it with the official locally vendored distribution before production use; the renderer interface is already isolated for that change.
