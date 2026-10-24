// Medición de latencia integral del navegador (SDD §16).
//
// El tiempo integral comienza al FINALIZAR la ventana de captura (antes de
// seleccionar y codificar JPEG) y termina cuando el resultado queda PRESENTADO
// en el navegador. Se usa performance.now() en ambos extremos del mismo
// navegador. La finalización de presentación se marca alrededor del render y el
// siguiente ciclo de pintura (requestAnimationFrame). No se afirma que el
// tiempo de fetch equivalga a tiempo hasta texto visible.

export class AttemptTimer {
  constructor() {
    this.captureEndMark = null; // fin de ventana de captura
    this.encodeStart = null;
    this.httpStart = null;
    this.httpEnd = null;
    this.renderStart = null;
    this.visibleMark = null; // texto visible (tras paint)
  }

  markCaptureEnd() {
    this.captureEndMark = performance.now();
  }
  markEncodeStart() {
    this.encodeStart = performance.now();
  }
  markHttpStart() {
    this.httpStart = performance.now();
  }
  markHttpEnd() {
    this.httpEnd = performance.now();
  }
  markRenderStart() {
    this.renderStart = performance.now();
  }

  // Marca la presentación tras el siguiente ciclo de pintura.
  async markVisibleAfterPaint() {
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve))
    );
    this.visibleMark = performance.now();
  }

  // Métricas derivadas, en milisegundos. null si falta algún extremo.
  summary() {
    const d = (a, b) => (a != null && b != null ? b - a : null);
    return {
      // Tiempo integral: fin de captura -> texto visible.
      integral_ms: d(this.captureEndMark, this.visibleMark),
      // Desgloses del cliente.
      select_encode_ms: d(this.encodeStart, this.httpStart),
      http_wait_ms: d(this.httpStart, this.httpEnd),
      visual_update_ms: d(this.renderStart, this.visibleMark),
    };
  }
}

// Acumula registros de intentos para exportación (reports/app/*.csv).
export class AttemptLog {
  constructor() {
    this.rows = [];
  }

  add(row) {
    this.rows.push(row);
  }

  toCsv() {
    const cols = [
      "attempt_id",
      "source",
      "mode",
      "api_version",
      "model_version",
      "feature_spec_hash",
      "capture_protocol_version",
      "integral_ms",
      "select_encode_ms",
      "http_wait_ms",
      "visual_update_ms",
      "server_total_ms",
      "decode_ms",
      "extraction_ms",
      "inference_ms",
      "decision_ms",
      "status",
      "reason_codes",
      "is_warmup",
      "timed_out",
    ];
    const esc = (v) => {
      if (v == null) return "";
      const s = Array.isArray(v) ? v.join("|") : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lines = [cols.join(",")];
    for (const r of this.rows) {
      lines.push(cols.map((c) => esc(r[c])).join(","));
    }
    return lines.join("\n");
  }

  download(filename = "latency_attempts.csv") {
    const blob = new Blob([this.toCsv()], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }
}
