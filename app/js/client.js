// Cliente HTTP del contrato lsc.http.v1 (SDD §9, §10).
//
// Construye el multipart/form-data con seis campos repetidos `frames`
// (frame_0.jpg..frame_5.jpg) y un campo de texto `metadata` (JSON). Tolera
// respuestas de infraestructura sin JSON. No reenvía automáticamente capturas
// fallidas: cada repetición requiere una acción visible del usuario.

export const SCHEMA_VERSION = "lsc.http.v1";

export function newClientRequestId() {
  if (crypto && crypto.randomUUID) return crypto.randomUUID();
  // Fallback simple si randomUUID no existe.
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export async function fetchConfig() {
  const res = await fetch("/api/v1/config", { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`config ${res.status}`);
  return res.json();
}

export async function fetchHealth() {
  const res = await fetch("/api/v1/health", { headers: { Accept: "application/json" } });
  let body = null;
  try {
    body = await res.json();
  } catch (_) {
    body = null;
  }
  return { status: res.status, body };
}

// Envía un intento. `sequence` proviene de capture.buildSequence().
// `protocolVersion` lo publica /config. Devuelve { httpStatus, envelope }.
export async function sendPredict(sequence, protocolVersion, options = {}) {
  const clientRequestId = options.clientRequestId || newClientRequestId();
  const metadata = {
    schema_version: SCHEMA_VERSION,
    client_request_id: clientRequestId,
    capture_protocol_version: protocolVersion,
    source: options.source || "webcam",
    width: sequence.width,
    height: sequence.height,
    frame_count: 6,
    observed_frame_count: sequence.observed,
    selected_indices: sequence.selectedIndices,
    timestamps_ms: sequence.frames.map((f) => f.timestampMs),
    capture_duration_ms: sequence.durationMs,
    mirrored: false,
  };

  const form = new FormData();
  sequence.frames.forEach((f, i) => {
    form.append("frames", f.blob, `frame_${i}.jpg`);
  });
  form.append("metadata", JSON.stringify(metadata));

  const controller = new AbortController();
  const timeoutMs = options.timeoutMs || 12000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let res;
  try {
    res = await fetch("/api/v1/predict", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }

  let envelope = null;
  try {
    envelope = await res.json();
  } catch (_) {
    // Respuesta de infraestructura sin JSON: la UI tolera este caso.
    envelope = null;
  }
  return { httpStatus: res.status, envelope, clientRequestId };
}
