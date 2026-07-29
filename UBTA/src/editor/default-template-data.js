// This ciphertext is deliberately shipped with the client so unlocking also works from file://.
export const defaultTemplateEnvelope = Object.freeze({
  version: 1,
  algorithm: 'AES-GCM',
  kdf: 'PBKDF2-SHA-256',
  iterations: 310000,
  salt: 'EG5326HC6gLkkN2TxPdodg==',
  iv: 'NyQQS8ak6Y2E3WUG',
  ciphertext:
    'VgziFV2P5dlFyYaBY3+oN5RpUx+0nqoK1jty5JHHz40YRcVuqNk0zaxzGO582YayTTI7XLl8LayWCeZx7y7yB71xE4/68pa1vcZCqg97tJI8bqRnsTDoZIJWdg0Sr4iFX+xWxYgMWDCtq6KHAabu+couoqS3qZ0bmdBBphyUMEHvPo1RLdQOLIyPE6+xa05GEON7iwUD8MSNsLfbwU7Y3MLuJu6+SpPdnvhu3oB2SmwT6ub+aRK3cJQKfoUffFni6dR9zgL1jJesPeM4xHCeoBpM3/1PPsYcpwccbsbX+Z7Hkd6XItPS1FPWn2UclHtsUSvE0FmgWgHpBb9uv43WpGgtsndet83Q483EeqGFfzUTdjaFg9keoIWdB4vJb1saQ0iwPekp9NPMKNPk8BdC2X4d8c827CH5jMEHHOBhOrEfedNERWEVpZmg6A==',
});
