// Historial de intentos en memoria de la pestaña (SDD §6).
//
// Máximo veinte intentos, SIN imágenes. Campos: número de intento, sugerencia,
// estado y corrección opcional. "Limpiar historial" los elimina; recargar
// también los borra (no se usa localStorage, IndexedDB ni cookies).
// Se conservan separadas la sugerencia original y la corrección del usuario.

export const HISTORY_LIMIT = 20;

export class History {
  constructor(limit = HISTORY_LIMIT) {
    this._items = [];
    this._limit = limit;
    this._counter = 0;
  }

  get items() {
    return this._items.slice();
  }

  nextAttemptNumber() {
    return this._counter + 1;
  }

  // Registra un intento. `correction` es null salvo que el usuario corrija.
  add({ status, suggestionLabelId, suggestionText, confidence }) {
    this._counter += 1;
    const entry = {
      attemptNumber: this._counter,
      status,
      // La sugerencia ORIGINAL del sistema no cambia nunca.
      prediction_original: {
        label_id: suggestionLabelId ?? null,
        text: suggestionText ?? null,
        confidence: confidence ?? null,
      },
      // La corrección del usuario se guarda aparte; no altera la predicción.
      correction: null,
    };
    this._items.unshift(entry);
    if (this._items.length > this._limit) {
      this._items.length = this._limit;
    }
    return entry;
  }

  // Aplica una corrección al intento más reciente. `labelId` null => "Ninguna".
  applyCorrection(attemptNumber, labelId, text) {
    const entry = this._items.find((e) => e.attemptNumber === attemptNumber);
    if (!entry) return;
    entry.correction = { label_id: labelId, text: text };
  }

  clear() {
    this._items = [];
  }
}
