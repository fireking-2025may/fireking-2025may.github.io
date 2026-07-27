# UBTA Steps Plan Editor

A client-side, desktop-first prototype for a constrained UBTA landscape steps plan. The serialisable schema is the source of truth; editor input is normalised to headings, paragraphs, three-level bullet/number lists, and text runs containing only review highlight and safe links.

## Development

Run `python3 -m http.server 8000 -d src`, then visit `http://localhost:8000`. No application server or network service is used by the editor.

The test suites are intentionally separate:

* Run `npm run test:schema` for the fast schema, validation, and history unit tests.
* Run `npm run test:editor` for DOM-oriented editor interaction and regression-fixture tests.

Run `npm test` to execute both suites.

## Standalone build

Run `npm run build`, then open `dist/index.html` directly in current Chrome or Edge (`file:///…/dist/index.html`). It contains all CSS and JavaScript and makes no runtime network requests. State and undo history remain in memory and disappear when the tab closes.

## Implemented scope

* Versioned closed schema for rich-text, image, and table blocks, with safe URL and image validation.
* Landscape A4 page preview with repeated report header, anchor-keyed linked contents, derived transaction proposals and `Page X of Y` footers.
* In-preview constrained editing that preserves sanitised highlights and links after every render.
* Editable step titles plus stable-ID add, reorder and delete controls.
* Editable accessible tables, highlighted total rows and keyboard-focusable column resizing.
* Validated PNG, JPEG, GIF and WebP uploads with required alternative text.

## Deliberately deferred

Diagrams, generated acronyms, appendix entities, optional sections, collaboration/workflow/version history, persistence/authentication/APIs, exports/imports, and mobile/tablet layouts are not present.

## Typography and supplied assets

No licensed Neuzeit Grotesk webfont was present in the repository, so the isolated font token uses `"Century Gothic", "Futura", "Aptos", "Segoe UI", sans-serif`. Exact type fidelity depends on licensed Neuzeit files being supplied.

The named reference attachments, including `image(51).png`, were not present in the workspace and the PNG is intentionally excluded from this draft pull request. The prototype therefore uses a compact text fallback in the repeated header. Integrating and embedding the authoritative header remains blocked until that supplied asset is available and approved for inclusion.

## Prototype limitation

The local `src/vendor/paged.js` preserves the upstream MIT attribution and supplies the narrow `Paged.Previewer` interface needed by this fixed prototype. The upstream Paged.js distribution could not be retrieved in this environment. Replace it with the official locally vendored distribution before production use; the renderer interface is already isolated for that change.
