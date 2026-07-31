# UBTA S77 contributor rules

- The runtime is offline-only and must make no network requests.
- Keep case data in memory only; never use browser storage, cookies, caches, or history state.
- Legal wording is fixed and may not be inferred, corrected, or rewritten.
- Use exact decimal arithmetic with deterministic half-up rounding; never binary floating point for values.
- Run lint, type checks, unit/integration tests, and browser tests before release.
- Email output in later steps must be classic Outlook `.msg`, never `.eml`.
- Every claim and release requires review and approval by Brook Smith.
