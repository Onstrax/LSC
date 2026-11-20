// Textos de la interfaz que dependen del resultado (sin DOM, para poder probarlos).
//
// Regla del proyecto: la probabilidad softmax NO es certeza. La interfaz presenta una
// SUGERENCIA que la persona confirma o corrige; el puntaje se muestra solo como dato
// del modelo, nunca como probabilidad de acierto.

export function retryMessage(codes) {
  const c = codes || [];
  // Calidad exige al menos una mano visible en 4 de 6 cuadros (no "ambas manos").
  if (c.includes("INSUFFICIENT_HANDS"))
    return "Mantenga al menos una mano a la vista durante toda la seña y repita.";
  if (c.includes("MULTIPLE_PEOPLE")) return "Deje una sola persona en el encuadre y repita.";
  if (c.includes("INSUFFICIENT_POSE")) return "Acerque rostro y hombros al encuadre y repita.";
  if (c.includes("HAND_ASSOCIATION_AMBIGUOUS"))
    return "Las manos se confunden entre sí. Sepárelas un poco y repita.";
  if (c.includes("GEOMETRY_OUTLIER"))
    return "La captura no se pudo medir con fiabilidad. Mejore la luz, mantenga el cuerpo dentro del encuadre y repita.";
  if (c.includes("INVALID_IMAGE") || c.includes("SIZE_MISMATCH") || c.includes("INVALID_FRAME_COUNT"))
    return "Los cuadros capturados no son válidos. Detenga y active la cámara de nuevo e intente otra vez.";
  if (c.includes("LOW_CONFIDENCE"))
    return "No hay una sugerencia clara. Repita la seña con calma.";
  return "La captura necesita repetirse. Ajuste el encuadre e intente de nuevo.";
}

export function errorMessage(httpStatus, codes) {
  const c = codes || [];
  if (c.includes("SERVICE_BUSY")) return "El servicio está ocupado. Espere un momento y reintente.";
  if (c.includes("MODEL_UNAVAILABLE") || c.includes("CONFIG_INCOMPATIBLE"))
    return "El servicio no está disponible. Avise al operador.";
  if (c.includes("INFERENCE_TIMEOUT")) return "El procesamiento tardó demasiado. Intente de nuevo.";
  return "Ocurrió un problema técnico. Intente de nuevo.";
}

// Puntaje máximo del modelo, con coma decimal. null si no es un número en [0, 1].
export function confidenceNote(confidence) {
  const v = Number(confidence);
  if (confidence == null || !Number.isFinite(v) || v < 0 || v > 1) return "";
  return (
    "Puntaje del modelo: " +
    v.toFixed(2).replace(".", ",") +
    ". No es la probabilidad de que la sugerencia sea correcta."
  );
}

// Nota bajo la sugerencia. Si la política lo pide (por defecto), se recuerda confirmar.
export function suggestionNote(vocabSize, requiresConfirmation = true) {
  const base = `Sugerencia dentro del vocabulario de ${vocabSize} señas. `;
  return requiresConfirmation
    ? base + "Confírmela o corríjala antes de usarla; no es una verificación de significado."
    : base + "No es una verificación de significado.";
}
