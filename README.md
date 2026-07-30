# UBTA S77 Letter Generator — Step 1

An offline-only, deterministic case-entry application for an eventual section 77 clearance letter workflow. Step 1 implements the in-memory data model, all nine form steps, validation, exact decimal calculations, navigation and clearing. It deliberately does **not** generate letters, PDFs, ZIP archives or Outlook messages.

## Develop and verify

Requires Node.js 20 or later. Dependencies are exactly pinned and recorded with integrity hashes in `package-lock.json`.

```sh
npm ci
npm run check
npm run test:browser
npm run build
```

Open `dist/UBTA-S77-Letter-Generator.html` directly in Google Chrome on Windows 11. The distributable is one self-contained file and has a restrictive Content Security Policy. It makes no network requests, uses no browser storage and retains the case only in memory. Reloading or closing it discards all entered information.

## Step 1 boundary

The Enclosures and Review pages are placeholders. The typed generation/import/export interfaces throw `Not implemented in this step`. PDF, ZIP, `.msg`, and bundle functionality belongs to later separately approved steps. Every eventual claim and release requires Brook Smith's review and approval.
