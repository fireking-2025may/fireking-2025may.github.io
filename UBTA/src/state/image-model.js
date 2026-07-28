const SUPPORTED_IMAGE_PATH = /\.(png|jpe?g|gif|webp)$/i;

export function validateImageSource(value) {
  const source = typeof value === 'string' ? value.trim() : '';
  if (!source) return { ok: false, error: 'missing-source' };

  let url;
  try {
    url = new URL(source);
  } catch {
    return { ok: false, error: 'malformed-source' };
  }

  if (url.protocol !== 'https:') return { ok: false, error: 'unsupported-protocol' };
  if (!SUPPORTED_IMAGE_PATH.test(url.pathname)) return { ok: false, error: 'unsupported-extension' };
  return { ok: true, url: url.href };
}

export function normalizeImageWidth(value) {
  return Math.min(100, Math.max(20, Number(value) || 100));
}

export function validateImageInput(input = {}) {
  const source = validateImageSource(input.source);
  if (!source.ok) return source;

  const altText = String(input.altText ?? '').trim();
  if (!altText) return { ok: false, error: 'missing-alt-text' };

  return {
    ok: true,
    image: {
      source: source.url,
      altText,
      width: normalizeImageWidth(input.width),
    },
  };
}
