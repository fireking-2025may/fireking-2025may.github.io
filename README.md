# Offline HS295 gift hold-over relief preparer

Open `hs295_holdover_relief_2026.html` directly in current Google Chrome or Microsoft Edge (`file://`); no server or installation is required. The final file is self-contained, includes the blank 2026 template internally, and makes no network requests.

## Scope and privacy

This prepares only the **HS295 Claim form 2026** for section 165 gifts of unlisted shares between individuals in one private trading company, under the fixed assumptions displayed in the application. It does not determine eligibility or perform statutory part-disposal apportionment. It assumes SP8/92 valuation deferment and produces no signatures or calculation schedule.

Nothing is automatically saved. Refreshing or closing loses the in-memory batch. Manual session JSON files are unencrypted and contain personal and tax-reference data; store and transmit them securely.

## Build and tests

Binary PDFs and rendered PNG fixtures are intentionally not stored in this repository so the change can be submitted through systems that reject binary files. The distributable HTML already contains the supplied template bytes.

To rebuild, obtain the exact supplied HS295 Claim form 2026 PDF locally and pass its path through `HS295_TEMPLATE`:

```sh
npm install --no-save pdf-lib@1.17.1 jszip@3.10.1 playwright@1.55.0
HS295_TEMPLATE='/local/path/HS295-Claim-form-2026.pdf' node hs295/build.mjs
node --test hs295/tests/*.test.js
node hs295/tests/browser.mjs
node hs295/tests/pdf-browser.mjs
```

The expected template SHA-256 is `c87ea3a00cfd9d1b84405d8f0570412e38c2e85d121b770bf56e697c640b4569`.

## Coordinates and future templates

`hs295/src/coordinates.js` documents every overlay target in PDF points with a bottom-left origin, Helvetica font choices, and the 7-point minimum. When HMRC releases a replacement, create a separately versioned build: verify page geometry and checksum, render the exact form locally at 144 DPI or above, recalibrate every coordinate, update capacities and schema labels, and rerun structural, visual, and browser tests.

## Manual Edge checklist

Open the HTML using `file://`; keep DevTools Network offline and confirm zero requests and no console errors. Create a batch, two reusable people, full and part-disposal transfers; edit, duplicate and delete; review and confirm; download each PDF and one ZIP; inspect that the ZIP contains PDFs only; export/import JSON after preview and confirmation; refresh and confirm all state is lost. Open each generated PDF and verify two A4 pages, overlay alignment, only the required two X marks, estimated-value E codes, and blank signatures/calculation-attached box.
