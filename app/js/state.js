// Máquina de estados central de la interfaz (SDD §7).
//
// Estados: idle, requesting_camera, ready, countdown, capturing, processing,
//          result, retry, error.
//
// Las transiciones se implementan aquí, no mediante banderas dispersas. Un
// identificador de generación evita que una respuesta tardía sustituya el
// resultado de otro intento. Durante countdown/capturing/processing no se
// acepta un segundo disparo.

export const STATES = Object.freeze({
  IDLE: "idle",
  REQUESTING_CAMERA: "requesting_camera",
  READY: "ready",
  COUNTDOWN: "countdown",
  CAPTURING: "capturing",
  PROCESSING: "processing",
  RESULT: "result",
  RETRY: "retry",
  ERROR: "error",
});

// Transiciones permitidas: estado -> conjunto de estados destino.
const TRANSITIONS = Object.freeze({
  [STATES.IDLE]: new Set([STATES.REQUESTING_CAMERA, STATES.ERROR]),
  [STATES.REQUESTING_CAMERA]: new Set([STATES.READY, STATES.IDLE, STATES.ERROR]),
  [STATES.READY]: new Set([STATES.COUNTDOWN, STATES.IDLE, STATES.ERROR]),
  [STATES.COUNTDOWN]: new Set([STATES.CAPTURING, STATES.READY, STATES.ERROR]),
  [STATES.CAPTURING]: new Set([STATES.PROCESSING, STATES.READY, STATES.ERROR]),
  [STATES.PROCESSING]: new Set([
    STATES.RESULT,
    STATES.RETRY,
    STATES.ERROR,
    STATES.READY,
  ]),
  [STATES.RESULT]: new Set([STATES.READY, STATES.ERROR, STATES.IDLE]),
  [STATES.RETRY]: new Set([STATES.READY, STATES.ERROR, STATES.IDLE]),
  [STATES.ERROR]: new Set([STATES.IDLE, STATES.READY, STATES.REQUESTING_CAMERA]),
});

// Estados en los que NO se admite un segundo disparo de captura.
export const BUSY_STATES = new Set([
  STATES.COUNTDOWN,
  STATES.CAPTURING,
  STATES.PROCESSING,
]);

export class StateMachine {
  constructor(onChange) {
    this._state = STATES.IDLE;
    this._generation = 0;
    this._onChange = onChange || (() => {});
  }

  get state() {
    return this._state;
  }

  get generation() {
    return this._generation;
  }

  // Avanza el identificador de generación: invalida respuestas de intentos
  // previos. Se llama al iniciar un intento y al cancelar.
  bumpGeneration() {
    this._generation += 1;
    return this._generation;
  }

  isGenerationCurrent(gen) {
    return gen === this._generation;
  }

  canCapture() {
    return this._state === STATES.READY;
  }

  isBusy() {
    return BUSY_STATES.has(this._state);
  }

  // Intenta transicionar. Lanza si la transición no está permitida, lo que
  // ayuda a detectar errores de lógica en pruebas.
  transition(next, payload) {
    const allowed = TRANSITIONS[this._state];
    if (!allowed || !allowed.has(next)) {
      throw new Error(
        `Transición inválida: ${this._state} -> ${next}`
      );
    }
    const prev = this._state;
    this._state = next;
    this._onChange(next, prev, payload);
    return next;
  }
}
