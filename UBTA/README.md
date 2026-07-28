# UBTA Steps Plan Editor

A client-side, desktop-first prototype for a constrained UBTA landscape steps plan. The serialisable schema is the source of truth; editor input is normalised to headings, paragraphs, three-level bullet/number lists, and text runs containing only review highlight and safe links.

## Development

Run `python3 -m http.server 8000 -d src`, then visit `http://localhost:8000`. No application server or network service is used by the editor. `npm run test:schema` runs schema, validation, persistence, and history tests; `npm run test:editor` runs rich-text interaction and pagination-scheduler tests; `npm test` runs both suites.

## Standalone build

Run `npm run build`, then open `dist/index.html` directly in current Chrome or Edge (`file:///…/dist/index.html`). It contains all CSS, JavaScript, and encrypted default-update data, so unlocking templates does not require a local fetch; only published images referenced by a document are fetched at runtime. Drafts, recovery data, and up to 20 named versions are stored only in that browser's local storage.

## Local persistence and snapshots

Edits autosave after a short idle period and are also saved when the page is hidden or closed. **Save now** performs the same operation immediately and, only after a successful save, replaces the current URL fragment with a snapshot of that saved state. Autosaves do not change the URL or add browser History entries. The previous draft is rotated into a recovery record. Saves use a revision check: if another tab has written a newer draft, the stale tab is refused before either the draft, recovery record, or URL is changed. Reload that tab to use the newer data.

Use **Save version** to create a named local checkpoint and **Versions** to restore or delete one; only the newest 20 are retained. A restore first saves the current draft and is cancelled if that prerequisite save fails. **Document properties** edits report metadata and sets workflow status to Draft, For review, or Final.

**Copy snapshot link** creates a schema-validated, versioned document snapshot in the URL fragment. Anyone opening the link receives an independent copy; later local edits are not synchronised. Snapshot contents are readable in the URL and must not be used for confidential client data. All persistence is browser-local: clearing site data, changing browser/profile, or using a different `file:` location can make it unavailable, so named versions are not a substitute for managed backups.

Headings are structural labels for the body content that follows them. Keep each heading immediately before its first related body paragraph (do not place another heading or an unrelated block between them). The paginator treats that heading and paragraph as a pair so a page break does not leave the heading stranded at the bottom of a page. Press **Enter** in a heading to create and focus the body paragraph directly below it.

Linked text remains directly editable. Click linked text to follow it, or place the caret in linked text and choose **Open link**. Internal links stay in the editor, while HTTP(S) destinations open in a separate protected tab.

## Implemented scope

* Versioned closed schema for rich-text, image, and table blocks, with safe URL and published-image validation.
* Browser-local autosave, recovery rotation, named versions, workflow status, and validated URL snapshots.
* Debounced, generation-gated preview pagination that preserves the active page, scroll position, focus, and text selection while rejecting stale renders.
* Landscape A4 page preview with repeated report header, anchor-keyed linked contents, derived transaction proposals and `Page X of Y` footers.
* In-preview constrained editing that preserves sanitised highlights and links after every render.
* Editable step and appendix titles plus stable-ID add, reorder and delete controls; derived alphabetical appendix labels update automatically.
* Editable accessible tables, highlighted total rows and keyboard-focusable column resizing.
* Published PNG, JPEG, GIF and WebP images referenced by validated HTTPS URLs, with required alternative text and optional captions and widths. Images remain hosted by their publisher and are not downloaded or base64-encoded into the document.

## Image compatibility

Image URLs must use HTTPS and end in `.png`, `.jpg`, `.jpeg`, `.gif` or `.webp` (a query string or fragment may follow). Because external images are referenced rather than embedded, they must remain available at their published URL for the preview to display them and may require an internet connection.

Schema version 4 no longer accepts `data:` URLs. Drafts, saved versions, recovery records and snapshots from earlier schema versions that contain base64-embedded images are rejected with migration guidance instead of silently losing those images. Publish each legacy image at a supported HTTPS URL and replace its `src` before importing that record. Earlier records containing only supported HTTPS image URLs continue to migrate normally.

## Deliberately deferred

Diagrams, generated acronyms, optional sections, collaboration, authentication/APIs, managed exports/backups, and mobile/tablet layouts are not present.

## Typography and supplied assets

No licensed Neuzeit Grotesk webfont was present in the repository, so the isolated font token uses `"Century Gothic", "Futura", "Aptos", "Segoe UI", sans-serif`. Exact type fidelity depends on licensed Neuzeit files being supplied.

The named reference attachments, including `image(51).png`, were not present in the workspace and the PNG is intentionally excluded from this draft pull request. The prototype therefore uses a compact text fallback in the repeated header. Integrating and embedding the authoritative header remains blocked until that supplied asset is available and approved for inclusion.

## Prototype limitation

The local `src/vendor/paged.js` preserves the upstream MIT attribution and supplies the narrow `Paged.Previewer` interface needed by this fixed prototype. The upstream Paged.js distribution could not be retrieved in this environment. Replace it with the official locally vendored distribution before production use; the renderer interface is already isolated for that change.
