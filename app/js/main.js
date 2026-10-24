// Orquestador de la interfaz (SDD §6, §7, §13, §16).
//
// Conecta la máquina de estados, la captura, el cliente HTTP, el historial y
// las métricas con el DOM. Mantiene mensajes accionables y accesibles.

import { StateMachine, STATES } from "./state.js";
import { CameraCapture, checkBrowserSupport, selectIndices } from "./capture.js";
import { fetchConfig, sendPredict } from "./client.js";
import { History } from "./history.js";
import { AttemptTimer, AttemptLog } from "./metrics.js";

const ui = {};
const attemptLog = new AttemptLog();
let config = null;
let camera = null;
let sm = null;
const history = new History();
let countdownTimer = null;

function $(id) {
  return document.getElementById(id);
}

function announce(msg) {
  // Región aria-live: anuncia cambios sin leer cada cuadro.
  ui.status.textContent = msg;
}

function setBanner(mode) {
  if (mode === "mock") {
    ui.banner.hidden = false;
    ui.banner.textContent = "Modo de prueba · respuestas simuladas";
  } else {
    ui.banner.hidden = true;
  }
}

function renderVocabulary(labels) {
  ui.vocabList.innerHTML = "";
  labels.forEach((label, i) => {
    const li = document.createElement("li");
    li.textContent = label;
    li.dataset.labelId = String(i);
    ui.vocabList.appendChild(li);
  });
  // Poblar el selector de corrección.
  ui.correctSelect.innerHTML = "";
  labels.forEach((label, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = label;
    ui.correctSelect.appendChild(opt);
  });
  const none = document.createElement("option");
  none.value = "none";
  none.textContent = "Ninguna de estas";
  ui.correctSelect.appendChild(none);
}

function renderHistory() {
  ui.historyList.innerHTML = "";
  for (const item of history.items) {
    const li = document.createElement("li");
    const orig = item.prediction_original.text || "—";
    let text = `#${item.attemptNumber} · ${item.status} · sugerencia: ${orig}`;
    if (item.correction) {
      text += ` · corrección: ${item.correction.text || "Ninguna"}`;
    }
    li.textContent = text;
    ui.historyList.appendChild(li);
  }
}

function showResultPanel(kind) {
  ui.resultAccepted.hidden = kind !== "accepted";
  ui.resultRetry.hidden = kind !== "retry";
  ui.resultError.hidden = kind !== "error";
  ui.correctionPanel.hidden = true;
}

// --- flujo principal -------------------------------------------------------
async function activateCamera() {
  if (sm.state !== STATES.IDLE && sm.state !== STATES.ERROR) return;
  const support = checkBrowserSupport();
  if (!support.supported) {
    sm.transition(STATES.ERROR);
    showResultPanel("error");
    ui.errorMsg.textContent =
      "Este navegador no es compatible (" + support.problems.join(", ") + ").";
    return;
  }
  sm.transition(STATES.REQUESTING_CAMERA);
  announce("Solicitando permiso de cámara…");
  camera = new CameraCapture(ui.video, config.capture_protocol);
  try {
    const dims = await camera.start();
    sm.transition(STATES.READY);
    announce(`Cámara activa (${dims.width}×${dims.height}). Encuadre rostro, hombros y manos.`);
    ui.captureBtn.disabled = false;
    ui.stopBtn.disabled = false;
  } catch (err) {
    sm.transition(STATES.ERROR);
    showResultPanel("error");
    if (err && err.name === "NotAllowedError") {
      ui.errorMsg.textContent = "Cámara denegada. Revise el permiso y reintente.";
    } else {
      ui.errorMsg.textContent = "No se pudo activar la cámara. Reconecte y active.";
    }
  }
}

function stopCamera() {
  if (camera) {
    camera.stop();
    camera = null;
  }
  clearTimeout(countdownTimer);
  ui.captureBtn.disabled = true;
  if (sm.state !== STATES.IDLE) {
    try {
      sm.transition(STATES.IDLE);
    } catch (_) {
      /* ya en idle */
    }
  }
  announce("Cámara detenida.");
}

async function captureSign() {
  // Un solo disparo: ignorar si no estamos en READY.
  if (!sm.canCapture()) return;
  const gen = sm.bumpGeneration();
  const timer = new AttemptTimer();

  sm.transition(STATES.COUNTDOWN);
  showResultPanel(null);
  let remaining = Math.round(config.capture_protocol.prep_ms / 1000);
  ui.countdown.hidden = false;
  ui.countdown.textContent = String(remaining);
  announce("Preparación: " + remaining);

  await new Promise((resolve) => {
    const tick = () => {
      if (sm.state !== STATES.COUNTDOWN || !sm.isGenerationCurrent(gen)) {
        resolve();
        return;
      }
      remaining -= 1;
      if (remaining <= 0) {
        resolve();
        return;
      }
      ui.countdown.textContent = String(remaining);
      announce("Preparación: " + remaining);
      countdownTimer = setTimeout(tick, 1000);
    };
    countdownTimer = setTimeout(tick, 1000);
  });

  if (sm.state !== STATES.COUNTDOWN || !sm.isGenerationCurrent(gen)) {
    ui.countdown.hidden = true;
    return;
  }

  // Captura: señal visual de "¡Ya!" durante la ventana.
  sm.transition(STATES.CAPTURING);
  ui.countdown.textContent = "¡Ya!";
  announce("Capturando la seña, ejecútela ahora.");
  const observed = await camera.captureWindow();
  timer.markCaptureEnd(); // fin de ventana de captura: inicia el reloj integral
  ui.countdown.hidden = true;

  if (!sm.isGenerationCurrent(gen) || document.hidden) {
    // Pestaña oculta o cancelado: descartar intento.
    try {
      sm.transition(STATES.READY);
    } catch (_) {}
    return;
  }

  timer.markEncodeStart();
  const sequence = await camera.buildSequence();
  if (!sequence.frames || sequence.frames.length < 6) {
    sm.transition(STATES.PROCESSING);
    sm.transition(STATES.RETRY);
    showResultPanel("retry");
    ui.retryMsg.textContent =
      `No se obtuvieron suficientes cuadros (${sequence.observed}/6). ` +
      "Repita manteniendo la seña dentro del encuadre.";
    announce(ui.retryMsg.textContent);
    returnToReady();
    return;
  }

  sm.transition(STATES.PROCESSING);
  announce("Procesando…");
  timer.markHttpStart();
  let response;
  try {
    response = await sendPredict(sequence, config.capture_protocol_version, {
      timeoutMs: config.capabilities.ui_wait_timeout_ms || 12000,
    });
  } catch (err) {
    timer.markHttpEnd();
    if (!sm.isGenerationCurrent(gen)) return;
    sm.transition(STATES.ERROR);
    showResultPanel("error");
    ui.errorMsg.textContent =
      err && err.name === "AbortError"
        ? "El servicio tardó demasiado. Intente de nuevo."
        : "No se pudo contactar el servicio. Verifique que esté activo e intente de nuevo.";
    announce(ui.errorMsg.textContent);
    recordAttempt(timer, { httpStatus: 0, envelope: null }, false);
    returnToReady();
    return;
  }
  timer.markHttpEnd();

  // Guardia de generación: una respuesta tardía no sustituye otro intento.
  if (!sm.isGenerationCurrent(gen)) return;

  await presentResponse(response, timer);
  recordAttempt(timer, response, false);
  returnToReady();
}

async function presentResponse(response, timer) {
  const { httpStatus, envelope } = response;
  timer.markRenderStart();

  if (!envelope) {
    sm.transition(STATES.ERROR);
    showResultPanel("error");
    ui.errorMsg.textContent =
      "Respuesta no comprensible del servicio. Intente de nuevo.";
    await timer.markVisibleAfterPaint();
    return;
  }

  if (envelope.status === "accepted") {
    sm.transition(STATES.RESULT);
    showResultPanel("accepted");
    ui.suggestionText.textContent = envelope.display_text;
    ui.vocabNote.textContent =
      "Sugerencia dentro del vocabulario de 10 señas. No es una verificación de significado.";
    const entry = history.add({
      status: "accepted",
      suggestionLabelId: envelope.label_id,
      suggestionText: envelope.display_text,
      confidence: envelope.confidence,
    });
    ui.resultAccepted.dataset.attempt = String(entry.attemptNumber);
    renderHistory();
    announce("Seña sugerida: " + envelope.display_text);
  } else if (envelope.status === "retry") {
    sm.transition(STATES.RETRY);
    showResultPanel("retry");
    ui.retryMsg.textContent = retryMessage(envelope.reason_codes);
    history.add({ status: "retry" });
    renderHistory();
    announce(ui.retryMsg.textContent);
  } else {
    sm.transition(STATES.ERROR);
    showResultPanel("error");
    ui.errorMsg.textContent = errorMessage(httpStatus, envelope.reason_codes);
    announce(ui.errorMsg.textContent);
  }
  await timer.markVisibleAfterPaint();
}

function retryMessage(codes) {
  const c = codes || [];
  if (c.includes("INSUFFICIENT_HANDS")) return "Muestre ambas manos y repita.";
  if (c.includes("MULTIPLE_PEOPLE")) return "Deje una sola persona en el encuadre y repita.";
  if (c.includes("INSUFFICIENT_POSE")) return "Acerque rostro y hombros al encuadre y repita.";
  if (c.includes("LOW_CONFIDENCE"))
    return "No hay una sugerencia clara. Repita la seña con calma.";
  return "La captura necesita repetirse. Ajuste el encuadre e intente de nuevo.";
}

function errorMessage(httpStatus, codes) {
  const c = codes || [];
  if (c.includes("SERVICE_BUSY")) return "El servicio está ocupado. Espere un momento y reintente.";
  if (c.includes("MODEL_UNAVAILABLE") || c.includes("CONFIG_INCOMPATIBLE"))
    return "El servicio no está disponible. Avise al operador.";
  if (c.includes("INFERENCE_TIMEOUT")) return "El procesamiento tardó demasiado. Intente de nuevo.";
  return "Ocurrió un problema técnico. Intente de nuevo.";
}

function returnToReady() {
  // Volver a ready si hay cámara; limpia countdown.
  clearTimeout(countdownTimer);
  ui.countdown.hidden = true;
  if (camera && (sm.state === STATES.RESULT || sm.state === STATES.RETRY || sm.state === STATES.PROCESSING)) {
    try {
      sm.transition(STATES.READY);
    } catch (_) {}
  }
}

function recordAttempt(timer, response, isWarmup) {
  const env = response.envelope || {};
  const t = (env.timings_ms) || {};
  const s = timer.summary();
  attemptLog.add({
    attempt_id: history.items[0] ? history.items[0].attemptNumber : "",
    source: "webcam",
    mode: (config && config.mode) || "",
    api_version: (config && config.api_version) || "",
    model_version: env.model_version || "",
    feature_spec_hash: env.feature_spec_hash || "",
    capture_protocol_version: env.capture_protocol_version || "",
    integral_ms: round(s.integral_ms),
    select_encode_ms: round(s.select_encode_ms),
    http_wait_ms: round(s.http_wait_ms),
    visual_update_ms: round(s.visual_update_ms),
    server_total_ms: round(t.server_total),
    decode_ms: round(t.decode),
    extraction_ms: round(t.extraction),
    inference_ms: round(t.inference),
    decision_ms: round(t.decision),
    status: env.status || (response.httpStatus === 0 ? "error" : ""),
    reason_codes: env.reason_codes || [],
    is_warmup: isWarmup ? "true" : "false",
    timed_out: response.httpStatus === 504 ? "true" : "false",
  });
}

function round(v) {
  return v == null ? null : Math.round(v * 100) / 100;
}

// --- corrección e historial ------------------------------------------------
function openCorrection() {
  ui.correctionPanel.hidden = false;
}

function applyCorrection() {
  const attempt = Number(ui.resultAccepted.dataset.attempt || "0");
  const val = ui.correctSelect.value;
  if (val === "none") {
    history.applyCorrection(attempt, null, "Ninguna");
  } else {
    const labelId = Number(val);
    history.applyCorrection(attempt, labelId, config.labels[labelId]);
  }
  ui.correctionPanel.hidden = true;
  renderHistory();
  announce("Corrección registrada (no cambia la sugerencia original).");
}

function confirmSuggestion() {
  // La confirmación NO cambia la predicción ni es etiqueta validada.
  announce("Confirmación registrada.");
  showResultPanel(null);
}

function clearHistory() {
  history.clear();
  renderHistory();
  announce("Historial eliminado.");
}

// --- inicialización --------------------------------------------------------
function cacheDom() {
  [
    "banner",
    "status",
    "vocabList",
    "video",
    "countdown",
    "activateBtn",
    "captureBtn",
    "stopBtn",
    "resultAccepted",
    "resultRetry",
    "resultError",
    "suggestionText",
    "vocabNote",
    "retryMsg",
    "errorMsg",
    "confirmBtn",
    "correctBtn",
    "repeatBtn",
    "correctionPanel",
    "correctSelect",
    "applyCorrectionBtn",
    "historyList",
    "clearHistoryBtn",
    "exportBtn",
  ].forEach((id) => {
    ui[id] = $(id);
  });
}

function wireEvents() {
  ui.activateBtn.addEventListener("click", activateCamera);
  ui.stopBtn.addEventListener("click", stopCamera);
  ui.captureBtn.addEventListener("click", captureSign);
  ui.confirmBtn.addEventListener("click", confirmSuggestion);
  ui.correctBtn.addEventListener("click", openCorrection);
  ui.applyCorrectionBtn.addEventListener("click", applyCorrection);
  ui.repeatBtn.addEventListener("click", () => {
    showResultPanel(null);
    returnToReady();
    announce("Listo para un nuevo intento.");
  });
  ui.clearHistoryBtn.addEventListener("click", clearHistory);
  ui.exportBtn.addEventListener("click", () => attemptLog.download());

  // Al ocultar la pestaña durante la captura, descartar el intento.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && sm && sm.isBusy()) {
      sm.bumpGeneration();
      clearTimeout(countdownTimer);
      ui.countdown.hidden = true;
      try {
        sm.transition(STATES.READY);
      } catch (_) {}
      announce("Intento descartado (pestaña oculta).");
    }
  });
}

async function init() {
  cacheDom();
  sm = new StateMachine();
  wireEvents();
  try {
    config = await fetchConfig();
  } catch (err) {
    showResultPanel("error");
    ui.errorMsg.textContent = "No se pudo cargar la configuración del servicio.";
    return;
  }
  setBanner(config.mode);
  renderVocabulary(config.labels);
  renderHistory();
  ui.captureBtn.disabled = true;
  ui.stopBtn.disabled = true;
  announce("Listo. Active la cámara para comenzar.");
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", init);
}

// Reexportación para pruebas unitarias.
export { selectIndices };
