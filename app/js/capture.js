// Captura reproducible en el navegador (SDD §8).
//
// - Sólo video, audio=false, resolución ideal 640x480, objetivo 30 FPS.
// - requestVideoFrameCallback para asociar cada muestra con un cuadro
//   presentado; se registra mediaTime y performance.now().
// - Vista previa espejada por CSS; los BYTES enviados NO van espejados.
// - Selección de seis índices: floor(k*(N-1)/5 + 0.5) para k=0..5.
// - Sin cuadros fabricados ni relleno; con N<6 se pide repetir.
// - Libera buffers, blobs y URLs tras envío/cancelación/error.

export const CAPTURE_DEFAULTS = Object.freeze({
  prepMs: 2000,
  windowMs: 1000,
  maxFps: 30,
  frameCount: 6,
  jpegQuality: 0.9,
  idealWidth: 640,
  idealHeight: 480,
});

// Verifica el contexto seguro y la disponibilidad de requestVideoFrameCallback.
export function checkBrowserSupport() {
  const problems = [];
  if (!window.isSecureContext && location.hostname !== "localhost") {
    problems.push("secure_context");
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    problems.push("getUserMedia");
  }
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
    problems.push("requestVideoFrameCallback");
  }
  return { supported: problems.length === 0, problems };
}

// Calcula los seis índices de muestreo. Exportada para pruebas unitarias.
// Fórmula exacta del SDD: floor(k*(N-1)/5 + 0.5), k=0..5.
export function selectIndices(n, frameCount = 6) {
  if (n < frameCount) {
    throw new Error(`N=${n} < ${frameCount}: no hay suficientes cuadros.`);
  }
  const idx = [];
  for (let k = 0; k < frameCount; k++) {
    idx.push(Math.floor((k * (n - 1)) / (frameCount - 1) + 0.5));
  }
  // Deben ser distintos y ordenados por construcción; comprobamos distinción.
  const distinct = new Set(idx);
  if (distinct.size !== frameCount) {
    throw new Error(`Índices no distintos para N=${n}: ${idx.join(",")}`);
  }
  return idx;
}

export class CameraCapture {
  constructor(videoEl, config = CAPTURE_DEFAULTS) {
    this._video = videoEl;
    this._cfg = { ...CAPTURE_DEFAULTS, ...config };
    this._stream = null;
    this._rvfcHandle = null;
    this._samples = []; // { mediaTime, perfNow, imageBitmap }
    this._cancelled = false;
    this._captureStart = 0;
    this._effectiveWidth = 0;
    this._effectiveHeight = 0;
  }

  get effectiveDimensions() {
    return { width: this._effectiveWidth, height: this._effectiveHeight };
  }

  async start() {
    // Solicitar sólo video. Las preferencias no son garantías.
    const constraints = {
      audio: false,
      video: {
        width: { ideal: this._cfg.idealWidth },
        height: { ideal: this._cfg.idealHeight },
        frameRate: { ideal: this._cfg.maxFps },
      },
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    // Si se canceló mientras resolvía, detener inmediatamente.
    if (this._cancelled) {
      stream.getTracks().forEach((t) => t.stop());
      throw new Error("cancelled");
    }
    this._stream = stream;
    this._video.srcObject = stream;
    await this._video.play().catch(() => {});

    const track = stream.getVideoTracks()[0];
    const settings = track ? track.getSettings() : {};
    this._effectiveWidth = settings.width || this._video.videoWidth;
    this._effectiveHeight = settings.height || this._video.videoHeight;
    return { width: this._effectiveWidth, height: this._effectiveHeight };
  }

  stop() {
    this._cancelled = true;
    if (this._rvfcHandle && "cancelVideoFrameCallback" in this._video) {
      this._video.cancelVideoFrameCallback(this._rvfcHandle);
      this._rvfcHandle = null;
    }
    if (this._stream) {
      this._stream.getTracks().forEach((t) => t.stop());
      this._stream = null;
    }
    this._video.srcObject = null;
    this._releaseSamples();
  }

  _releaseSamples() {
    for (const s of this._samples) {
      if (s.imageBitmap && s.imageBitmap.close) {
        s.imageBitmap.close();
      } else if (s.bmpPromise) {
        // Cerrar el bitmap cuando la promesa se resuelva, para no filtrar.
        s.bmpPromise.then((b) => b && b.close && b.close()).catch(() => {});
      }
    }
    this._samples = [];
  }

  // Acumula muestras durante la ventana de captura usando
  // requestVideoFrameCallback. Resuelve con el número de muestras observadas.
  async captureWindow() {
    this._cancelled = false;
    this._samples = [];
    // Medimos el transcurso con performance.now() PROPIO. El primer argumento
    // de requestVideoFrameCallback no siempre comparte base con performance.now
    // entre navegadores, así que no lo usamos para cerrar la ventana.
    this._captureStart = performance.now();
    const minInterval = 1000 / this._cfg.maxFps; // separación temporal mínima
    let lastMediaTime = -1;
    let lastSampleAt = -Infinity;
    const maxSamples = Math.ceil((this._cfg.windowMs / 1000) * this._cfg.maxFps) + 4;

    return new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        resolve(this._samples.length);
      };

      const onFrame = (_rvfcNow, metadata) => {
        if (this._cancelled) {
          finish();
          return;
        }
        const nowPerf = performance.now();
        const elapsed = nowPerf - this._captureStart;
        if (elapsed >= this._cfg.windowMs) {
          finish();
          return;
        }
        // mediaTime identifica el cuadro presentado; sirve para no duplicar.
        const mediaTime = metadata && typeof metadata.mediaTime === "number"
          ? metadata.mediaTime
          : nowPerf / 1000;
        const separated = nowPerf - lastSampleAt >= minInterval - 1;
        if (mediaTime !== lastMediaTime && separated) {
          lastMediaTime = mediaTime;
          lastSampleAt = nowPerf;
          // Capturamos el fotograma de forma SÍNCRONA al cuadro (sin await que
          // consuma la ventana): createImageBitmap devuelve una promesa que
          // guardamos; se resuelve en paralelo y se espera en buildSequence.
          const bmpPromise = createImageBitmap(this._video).catch(() => null);
          this._samples.push({ mediaTime, perfNow: nowPerf, bmpPromise });
        }
        if (this._samples.length >= maxSamples) {
          finish();
          return;
        }
        this._rvfcHandle = this._video.requestVideoFrameCallback(onFrame);
      };
      this._rvfcHandle = this._video.requestVideoFrameCallback(onFrame);
      // Salvaguarda por tiempo absoluto, por si rVFC deja de emitir.
      setTimeout(finish, this._cfg.windowMs + 300);
    });
  }

  // Selecciona seis cuadros y los codifica como JPEG sin espejo.
  // Devuelve { frames: [{ blob, index, timestampMs }], observed, durationMs }.
  async buildSequence() {
    // Resolver los bitmaps capturados durante la ventana. Descartar los que
    // fallaron (null) sin fabricar cuadros.
    await Promise.all(
      this._samples.map(async (s) => {
        if (s.bmpPromise && !s.imageBitmap) {
          s.imageBitmap = await s.bmpPromise;
        }
      })
    );
    // Quedarnos solo con muestras que produjeron un bitmap real.
    this._samples = this._samples.filter((s) => s.imageBitmap);

    const observed = this._samples.length;
    if (observed < this._cfg.frameCount) {
      this._releaseSamples();
      return { frames: null, observed, durationMs: 0 };
    }
    const indices = selectIndices(observed, this._cfg.frameCount);
    const t0 = this._samples[0].perfNow;
    // Duración real de la ventana: desde el primer al último cuadro muestreado.
    // Debe cubrir el último timestamp seleccionado (el servidor lo valida).
    const lastPerf = this._samples[observed - 1].perfNow;
    const durationMs = Math.max(this._cfg.windowMs, lastPerf - t0);

    // Conservar índice y tiempo ANTES de operaciones asíncronas; Promise.all
    // respeta el orden de entrada.
    const selected = indices.map((i) => ({
      sample: this._samples[i],
      selectedIndex: i,
    }));

    const canvas = document.createElement("canvas");
    canvas.width = this._effectiveWidth;
    canvas.height = this._effectiveHeight;
    const ctx = canvas.getContext("2d");

    const encodeOne = (entry) =>
      new Promise((resolve, reject) => {
        // Dibujar el bitmap SIN espejo (sin transformaciones).
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(entry.sample.imageBitmap, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error("toBlob falló"));
              return;
            }
            resolve({
              blob,
              index: entry.selectedIndex,
              timestampMs: entry.sample.perfNow - t0,
            });
          },
          "image/jpeg",
          this._cfg.jpegQuality
        );
      });

    const frames = await Promise.all(selected.map(encodeOne));
    this._releaseSamples();
    return {
      frames,
      observed,
      durationMs,
      selectedIndices: indices,
      width: this._effectiveWidth,
      height: this._effectiveHeight,
    };
  }
}
