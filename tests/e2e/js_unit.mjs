// Pruebas unitarias en Node de la lógica pura de los módulos JS.
//
// Cubren la fórmula de muestreo (U03), la máquina de estados (§7), el
// historial (§6) y las métricas (§16) sin necesidad de un navegador. Se
// ejecutan con el runner integrado de Node:
//
//     node --test tests/e2e/js_unit.mjs
//
// Estas pruebas complementan las e2e de Playwright (que requieren navegador).

import { test } from "node:test";
import assert from "node:assert/strict";

import { selectIndices } from "../../app/js/capture.js";
import { StateMachine, STATES } from "../../app/js/state.js";
import { History } from "../../app/js/history.js";
import { AttemptLog } from "../../app/js/metrics.js";

// --- U03: fórmula de muestreo ---------------------------------------------
test("selectIndices N=6 -> 0..5", () => {
  assert.deepEqual(selectIndices(6), [0, 1, 2, 3, 4, 5]);
});

test("selectIndices N=30 -> 0,6,12,17,23,29", () => {
  assert.deepEqual(selectIndices(30), [0, 6, 12, 17, 23, 29]);
});

test("selectIndices N<6 lanza", () => {
  assert.throws(() => selectIndices(5));
});

test("selectIndices produce índices distintos y crecientes para varios N", () => {
  for (let n = 6; n <= 120; n++) {
    const idx = selectIndices(n);
    assert.equal(idx.length, 6);
    for (let k = 1; k < idx.length; k++) {
      assert.ok(idx[k] > idx[k - 1], `N=${n} no creciente: ${idx}`);
    }
    assert.ok(idx[5] <= n - 1);
  }
});

// --- §7: máquina de estados ------------------------------------------------
test("transición válida idle -> requesting_camera", () => {
  const sm = new StateMachine();
  sm.transition(STATES.REQUESTING_CAMERA);
  assert.equal(sm.state, STATES.REQUESTING_CAMERA);
});

test("transición inválida lanza", () => {
  const sm = new StateMachine();
  assert.throws(() => sm.transition(STATES.CAPTURING));
});

test("busy states bloquean un segundo disparo", () => {
  const sm = new StateMachine();
  sm.transition(STATES.REQUESTING_CAMERA);
  sm.transition(STATES.READY);
  sm.transition(STATES.COUNTDOWN);
  assert.ok(sm.isBusy());
  assert.ok(!sm.canCapture());
});

test("generación invalida respuestas tardías", () => {
  const sm = new StateMachine();
  const g1 = sm.bumpGeneration();
  const g2 = sm.bumpGeneration();
  assert.ok(!sm.isGenerationCurrent(g1));
  assert.ok(sm.isGenerationCurrent(g2));
});

// --- §6: historial ---------------------------------------------------------
test("historial limita a 20 y separa corrección de predicción", () => {
  const h = new History(20);
  for (let i = 0; i < 25; i++) {
    h.add({ status: "accepted", suggestionLabelId: 0, suggestionText: "Hola", confidence: 0.9 });
  }
  assert.equal(h.items.length, 20);
  const last = h.items[0];
  h.applyCorrection(last.attemptNumber, 1, "Buenos");
  assert.equal(last.prediction_original.text, "Hola"); // no cambia
  assert.equal(last.correction.text, "Buenos"); // separada
});

test("limpiar historial lo vacía", () => {
  const h = new History();
  h.add({ status: "retry" });
  h.clear();
  assert.equal(h.items.length, 0);
});

// --- §16: CSV de métricas --------------------------------------------------
test("AttemptLog genera CSV con cabecera de unidades y null != cero", () => {
  const log = new AttemptLog();
  log.add({ attempt_id: 1, status: "accepted", integral_ms: 1234.5, decode_ms: null });
  const csv = log.toCsv();
  const [header, row] = csv.split("\n");
  assert.ok(header.includes("integral_ms"));
  const cols = header.split(",");
  const vals = row.split(",");
  const decodeIdx = cols.indexOf("decode_ms");
  // null se exporta como celda vacía, distinta de "0".
  assert.equal(vals[decodeIdx], "");
});
