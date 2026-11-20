// Referencias visuales de cómo se hace cada seña (vocabulario admitido).
//
// Los archivos viven en app/assets/signs/ y se describen en manifest.json, indexado por el
// texto de la etiqueta (el mismo que publica /api/v1/config). Sin dependencias del DOM para
// poder probarlo. Si falta el manifiesto, una entrada o un archivo, la interfaz sigue
// funcionando y avisa que la referencia no está disponible: nunca se inventa una.

export const SIGNS_BASE = "/static/assets/signs/";

// Solo nombres de archivo simples dentro de la carpeta: sin rutas, sin "..", sin URLs externas.
const SAFE_FILE = /^[A-Za-z0-9_-]+\.(png|jpe?g|webp|gif|svg|mp4|webm)$/i;
const VIDEO = /\.(mp4|webm)$/i;

export async function loadSignManifest(fetchFn = fetch) {
  try {
    const res = await fetchFn(SIGNS_BASE + "manifest.json", { headers: { Accept: "application/json" } });
    if (!res.ok) return {};
    const doc = await res.json();
    return doc && typeof doc.signs === "object" && doc.signs ? doc.signs : {};
  } catch (_) {
    return {};
  }
}

function cleanText(v) {
  return typeof v === "string" && v.trim() ? v.trim() : null;
}

// Devuelve la referencia de una seña, ya validada, para que el DOM solo la pinte.
//   { kind: "image"|"video"|null, src, alt, tip, credit }
export function signReference(signs, label) {
  const entry = signs && Object.prototype.hasOwnProperty.call(signs, label) ? signs[label] : null;
  const ref = { kind: null, src: null, alt: null, tip: null, credit: null };
  if (!entry || typeof entry !== "object") return ref;
  ref.tip = cleanText(entry.tip);
  ref.credit = cleanText(entry.credit);
  const file = cleanText(entry.file);
  if (file && SAFE_FILE.test(file)) {
    ref.kind = VIDEO.test(file) ? "video" : "image";
    ref.src = SIGNS_BASE + file;
    ref.alt = cleanText(entry.alt) || `Cómo se hace la seña «${label}»`;
  }
  return ref;
}

// Texto cuando no hay archivo para la seña elegida.
export function missingReferenceText(label) {
  return `Aún no hay una referencia visual para «${label}». Pida el apoyo de una persona que conozca LSC.`;
}
