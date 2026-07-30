# Offline HS295 gift hold-over relief preparer

Open `hs295_holdover_relief_2026.html` directly in current Google Chrome or Microsoft Edge (`file://`); no server or installation is required. The file is self-contained and makes no network requests. Chrome is automated in this repository; Edge must be checked manually where unavailable.

## Scope and privacy

This prepares only the **HS295 Claim form 2026** for section 165 gifts of unlisted shares between individuals in one private trading company, under the fixed assumptions displayed in the application. It does not determine eligibility or perform statutory part-disposal apportionment. It assumes SP8/92 valuation deferment and produces no signatures or calculation schedule.

Nothing is automatically saved. Refreshing or closing loses the in-memory batch. Manual session JSON files are unencrypted and contain personal and tax-reference data; store and transmit them securely.

## Build and tests

```sh
npm install --no-save pdf-lib@1.17.1 jszip@3.10.1 playwright@1.55.0
node hs295/build.mjs
node --test hs295/tests/*.test.js
node hs295/tests/browser.mjs
```

The immutable template is `hs295/HS295-Claim-form-2026.pdf`, copied from the supplied `UBTA/src/img/HS295-Claim-form-2026 empty.pdf`. SHA-256: `c87ea3a00cfd9d1b84405d8f0570412e38c2e85d121b770bf56e697c640b4569`.

## Coordinates and future templates

`hs295/src/coordinates.js` documents every overlay target in PDF points with a bottom-left origin, Helvetica font choices, and the 7-point minimum. When HMRC releases a replacement, never overwrite the 2026 build: copy the new PDF, verify page geometry and checksum, inspect/render at 144 DPI or above, recalibrate every coordinate against that exact form, update capacities and version/schema labels, regenerate fixtures, and rerun structural, visual, and browser tests.

## Manual Edge checklist

Open the HTML using `file://`; keep DevTools Network offline and confirm zero requests and no console errors. Create a batch, two reusable people, full and part-disposal transfers; edit, duplicate and delete; review and confirm; download each PDF and one ZIP; inspect the ZIP contains PDFs only; export/import JSON after preview and confirmation; refresh and confirm all state is lost. Open each PDF and verify two A4 pages, overlay alignment, only the required two X marks, estimated-value E codes, and blank signatures/calculation-attached box.
