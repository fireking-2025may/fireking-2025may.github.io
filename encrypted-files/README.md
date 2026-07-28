# Encrypted text vault

Open `index.html` through GitHub Pages (or another HTTPS/static web server) and enter the separately shared vault password. All key derivation and decryption happen in the browser with the Web Crypto API. The password is not stored in this repository.

Files under `vault/` are JSON text envelopes containing only a random salt, IV, algorithm metadata, and AES-256-GCM ciphertext. The decrypted JSON payload has this shape:

```json
{ "name": "example.txt", "content": "Private text goes here." }
```

Each file uses a unique 16-byte salt, a unique 12-byte IV, and 310,000 PBKDF2-SHA-256 iterations. The filenames inside each payload are encrypted; `manifest.json` exposes only the opaque storage paths needed by the static page.

## Important limitations

- Serve the directory over HTTPS or `localhost`; browsers may disable Web Crypto or `fetch()` on `file://` pages.
- Anyone can download the ciphertext from a public GitHub repository and attempt passwords offline. Use a long, randomly generated password and share it outside the repository.
- Git history retains old encrypted files. Rotate the password and re-encrypt every file if the password is exposed.
- This page protects file contents at rest; it does not provide user accounts, access revocation, or protection from malicious changes to the hosted JavaScript.
