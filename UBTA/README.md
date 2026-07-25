# UBTA Steps Plan Editor — phases 1–5 prototype

A client-side, desktop-first editor for a constrained UBTA landscape steps plan. Its versioned, serialisable schema is the source of truth; input is normalised before every committed change.

## Development and tests

Run `python3 -m http.server 8000 -d src`, then visit `http://localhost:8000` in current Chrome or Edge. Run `npm test` for the schema, migration, derived-content, structural-editing, DOM parsing, renderer, and history suites. The tests deliberately exercise pure modules without requiring a browser.

Run `npm run build` to create `dist/index.html`. The generated file embeds the same schema, helpers, controls, CSS, application code, and local Paged preview shim as `src/`; it makes no runtime network requests. State and undo history remain in memory and disappear when the tab closes.

## Implemented scope

### Phases 1–3

* A closed, versioned schema, neutral seed document, fixed cover, repeated report header, stable anchors, linked contents and page footers.
* Constrained rich text: four heading levels, paragraphs, review highlighting, safe HTTP(S), mail and internal links, bullet/number lists up to three levels, plain-text paste, and debounced reflow.
* Page navigation and session-only undo/redo.

### Phase 4 — structured content

* Normalised semantic tables with bounded column widths, optional configured totals, and drag handles for column sizing.
* Embedded PNG, JPEG, GIF and WebP data/blob images with required safe sources, alt text, optional captions, and bounded rendered dimensions.
* A completed column or image resize gesture creates one undo entry, rather than one entry for each pointer movement.

### Phase 5 — steps and derived content

* Add, duplicate, delete, and move-step controls operate on stable step IDs; visible step numbers are always derived from current order and are not identity.
* Contents anchors and generated proposal links remain stable after reorder/delete, while labels and proposal numbers are regenerated. A step's explicit proposal overrides its first paragraph; otherwise that paragraph supplies the proposal text.
* Structural edits participate in the same undo/redo history as content edits.

## Limits

This prototype has no collaboration, durable workflow/version history, persistence, authentication, server API, import/export format, automatic numeric table calculation, image cropping, appendix entity editor, or mobile/tablet layout. Table totals identify configured total columns but do not evaluate arbitrary formulae. Images must already be local data/blob sources; remote image URLs are rejected. Pagination uses the narrow locally vendored `Paged.Previewer` shim, not the full upstream distribution.

No licensed Neuzeit Grotesk webfont or authoritative supplied header image was present, so the editor uses the documented fallback font stack and a compact text header. Replace these only when approved assets are available.
