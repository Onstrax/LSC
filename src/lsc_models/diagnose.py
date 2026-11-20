"""Diagnóstico de intentos REALES volcados por la app (``api/debug_dump.py``).

Uso:  python -m src.lsc_models diagnose-dump --dir artifacts/debug_dump --expect "Hola*3,Buenos*3,..."

Para cada intento (en orden cronológico) muestra qué predijo el modelo y con qué confianza, cuántos
cuadros tenían cada mano, y compara con el corpus de entrenamiento:

- presencia de mano izquierda/derecha por cuadro (bundle ``features.npz``, solo ``train`` procesable);
- tamaño de los hombros respecto al ancho de la imagen (muestra de cuadros del corpus, si está en disco);
- y si **espejar la entrada** (x -> -x e izquierda<->derecha) sube la confianza, lo que delataría lados invertidos.

No modifica el modelo, los datos ni el split ``test``. Las capturas son evidencia de la APP, no del corpus.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

SPEC = Path(__file__).resolve().parents[2] / "contracts" / "feature_spec.json"
LABELS = Path(__file__).resolve().parents[2] / "contracts" / "labels.json"
N_POINTS = 49


# --- utilidades puras (con pruebas) -------------------------------------------------------------
def point_index() -> dict[str, int]:
    order = json.loads(SPEC.read_text(encoding="utf-8"))["point_order"]
    return {name: i for i, name in enumerate(order)}


def mirror_features(x: np.ndarray, *, flip_x: bool = True, swap_sides: bool = True) -> np.ndarray:
    """Imagen espejada: x -> -x en puntos presentes e intercambio de lados (pose y manos).

    Con x normalizada respecto al punto medio de los hombros, espejar la imagen equivale a esto.
    """
    idx = point_index()
    z = np.asarray(x, dtype=np.float32).reshape(-1, N_POINTS, 3).copy()
    if flip_x:
        present = z[..., 2] > 0
        z[..., 0] = np.where(present, -z[..., 0], z[..., 0])
    if swap_sides:
        for a, b in (("pose_11", "pose_12"), ("pose_13", "pose_14"), ("pose_15", "pose_16")):
            z[:, [idx[a], idx[b]], :] = z[:, [idx[b], idx[a]], :]
        left = [idx[f"lh_{i}"] for i in range(21)]
        right = [idx[f"rh_{i}"] for i in range(21)]
        z[:, left + right, :] = z[:, right + left, :]
    return z.reshape(np.asarray(x).shape)


def hand_presence(x: np.ndarray) -> tuple[int, int]:
    """Cuadros (de 6) con alguna mano izquierda / derecha presente en ``x``."""
    idx = point_index()
    z = np.asarray(x, dtype=np.float32).reshape(-1, N_POINTS, 3)
    left = (z[:, [idx[f"lh_{i}"] for i in range(21)], 2] > 0).any(axis=1)
    right = (z[:, [idx[f"rh_{i}"] for i in range(21)], 2] > 0).any(axis=1)
    return int(left.sum()), int(right.sum())


def parse_expected(spec: str | None) -> list[str]:
    """``"Hola*3,Buenos,Días*2"`` -> ``["Hola","Hola","Hola","Buenos","Días","Días"]``."""
    out: list[str] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, times = part.partition("*")
        out.extend([name.strip()] * (int(times) if times.strip() else 1))
    return out


def load_attempts(dump_dir: Path) -> list[dict]:
    """Intentos volcados, en orden cronológico (el nombre de carpeta empieza por la fecha)."""
    attempts = []
    for d in sorted(p for p in Path(dump_dir).iterdir() if p.is_dir() and (p / "meta.json").is_file()):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        x = np.load(d / "x.npy") if (d / "x.npy").is_file() else None
        attempts.append({"dir": d, "meta": meta, "x": x})
    return attempts


# --- referencias del corpus ---------------------------------------------------------------------
def corpus_hand_rates(bundle_path: Path) -> tuple[float, float] | None:
    """Fracción de cuadros con mano izquierda/derecha en train (procesables). ``None`` si no hay bundle."""
    try:
        b = np.load(bundle_path, allow_pickle=False)
        m = (b["split"] == "train") & b["quality_pass"]
        counts = np.array([hand_presence(x) for x in b["X"][m]], dtype=np.float64)
        return float(counts[:, 0].mean() / 6), float(counts[:, 1].mean() / 6)
    except Exception:
        return None


def _detector(artifacts: Path):
    from src.lsc_features.detector import RealDetector

    return RealDetector(artifacts / "pose_landmarker.task", artifacts / "hand_landmarker.task")


def _quality_kwargs() -> dict:
    rules = json.loads(SPEC.read_text(encoding="utf-8"))["quality_rules"]
    return dict(
        min_shoulder_distance_frac_width=float(rules["min_shoulder_distance_frac_width"]),
        visibility_threshold=float(rules["visibility_threshold"]),
        presence_threshold=float(rules["presence_threshold"]),
        use_presence=bool(rules["use_presence"]),
    )


def _frame_stats(detector, rgb: np.ndarray) -> tuple[int, float | None]:
    """(manos detectadas por MediaPipe en el cuadro, distancia de hombros / ancho o None)."""
    from src.lsc_features import geometry as geo

    pose = detector.detect_pose(rgb)
    hands = detector.detect_hands(rgb)
    persons = list(getattr(pose, "pose_landmarks", []) or [])
    n_hands = len(list(getattr(hands, "hand_landmarks", []) or []))
    if not persons:
        return n_hands, None
    info = geo.shoulder_midpoint_and_distance(persons[0], rgb.shape[1], rgb.shape[0], **_quality_kwargs())
    return n_hands, (None if info is None else info[1] / rgb.shape[1])


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def corpus_shoulder_frac(detector, raw_dir: Path | None, n: int = 30, seed: int = 0) -> float | None:
    """Mediana de (distancia de hombros / ancho) en ``n`` cuadros del corpus. ``None`` si no está en disco."""
    try:
        from .prepare import resolve_raw_dir

        raw = resolve_raw_dir(raw_dir)
        files = sorted(raw.glob("Per*/*/*.jpg"))
        if not files:
            return None
        rng = random.Random(seed)
        vals = []
        for f in rng.sample(files, min(n, len(files))):
            _, frac = _frame_stats(detector, _read_rgb(f))
            if frac is not None:
                vals.append(frac)
        return float(np.median(vals)) if vals else None
    except Exception:
        return None


# --- comando ------------------------------------------------------------------------------------
def run_diagnose(dump_dir: str, artifacts: str, expect: str | None, *, redetect: bool = True,
                 bundle: str | None = None, raw_dir: str | None = None, corpus_sample: int = 30) -> int:
    dump, art = Path(dump_dir), Path(artifacts)
    if not dump.is_dir():
        print(f"No existe {dump}. Arranque la app con `run-app --mode real --debug-dump` y haga intentos.", file=sys.stderr)
        return 3
    attempts = load_attempts(dump)
    if not attempts:
        print(f"{dump} no contiene intentos (carpetas con meta.json).", file=sys.stderr)
        return 3

    labels = [e["display_text"] for e in sorted(json.loads(LABELS.read_text(encoding="utf-8"))["labels"], key=lambda e: e["class_id"])]
    expected = parse_expected(expect)
    if expected and len(expected) != len(attempts):
        print(f"AVISO: --expect tiene {len(expected)} entradas y hay {len(attempts)} intentos; se alinean por orden "
              "y los sobrantes quedan sin esperado.", file=sys.stderr)
    thr = None
    try:
        thr = float(json.loads((art / "thresholds.json").read_text(encoding="utf-8"))["threshold"])
    except Exception:
        pass

    from src.lsc_inference import Predictor

    model = Predictor(art / "model.keras")
    det = _detector(art) if redetect else None

    rows = []
    for i, a in enumerate(attempts):
        meta, x = a["meta"], a["x"]
        row = {"n": i + 1, "carpeta": a["dir"].name, "esperada": expected[i] if i < len(expected) else "",
               "calidad": bool(meta.get("quality_pass")), "razones": " ".join(meta.get("reason_codes", [])),
               "top1": "", "conf": "", "top2": "", "manos_izq": "", "manos_der": "", "manos_det_media": "",
               "hombros_frac": "", "espejo_top1": "", "espejo_conf": ""}
        if x is not None:
            l, r = hand_presence(x)
            row["manos_izq"], row["manos_der"] = l, r
            if meta.get("quality_pass"):
                p = np.asarray(model.predict(x[None, ...]), dtype=np.float32).reshape(-1)
                order = np.argsort(p)[::-1]
                row.update(top1=labels[int(order[0])], conf=round(float(p[order[0]]), 3),
                           top2=f"{labels[int(order[1])]} {p[order[1]]:.2f}")
                pm = np.asarray(model.predict(mirror_features(x)[None, ...]), dtype=np.float32).reshape(-1)
                row.update(espejo_top1=labels[int(pm.argmax())], espejo_conf=round(float(pm.max()), 3))
        if det is not None:
            stats = [_frame_stats(det, _read_rgb(a["dir"] / f"frame_{k}.jpg")) for k in range(6)]
            row["manos_det_media"] = round(float(np.mean([s[0] for s in stats])), 2)
            fr = [s[1] for s in stats if s[1] is not None]
            row["hombros_frac"] = round(float(np.median(fr)), 3) if fr else ""
        rows.append(row)

    cols = ["n", "esperada", "top1", "conf", "top2", "calidad", "razones", "manos_izq", "manos_der",
            "manos_det_media", "hombros_frac", "espejo_top1", "espejo_conf", "carpeta"]
    print("\nIntento | esperada -> top1 (conf) | top2 | manos izq/der (de 6) | detectadas/cuadro | hombros/ancho | espejo")
    for r in rows:
        print(f"{r['n']:>3} | {r['esperada'] or '-':8s} -> {r['top1'] or '(rechazado)':8s} ({r['conf'] if r['conf'] != '' else '--'}) | "
              f"{r['top2'] or '-':14s} | {r['manos_izq']}/{r['manos_der']} | {r['manos_det_media']} | {r['hombros_frac']} | "
              f"{r['espejo_top1'] or '-'} ({r['espejo_conf'] if r['espejo_conf'] != '' else '--'})  {r['razones']}")
    out_csv = dump / "diagnosis.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # --- resumen ---
    ok = [r for r in rows if r["conf"] != ""]
    print(f"\nResumen ({len(rows)} intentos; {len(ok)} pasaron calidad; CSV: {out_csv})")
    if ok:
        confs = np.array([float(r["conf"]) for r in ok])
        print(f"  confianza top-1: mediana {np.median(confs):.3f}, mín {confs.min():.3f}, máx {confs.max():.3f}")
        if thr is not None:
            print(f"  superan el umbral {thr:.3f}: {(confs >= thr).sum()} de {len(ok)} ({(confs >= thr).mean():.0%})  "
                  "[en validación del corpus: 81 %]")
        withexp = [r for r in ok if r["esperada"]]
        if withexp:
            hit = np.array([r["top1"] == r["esperada"] for r in withexp])
            print(f"  top-1 correcto: {hit.sum()} de {len(withexp)} ({hit.mean():.0%})")
        better = [r for r in ok if r["espejo_conf"] != "" and float(r["espejo_conf"]) > float(r["conf"])]
        print(f"  espejar la entrada SUBE la confianza en {len(better)} de {len(ok)} intentos "
              "(si es la mayoría, sospeche de lados invertidos)")
    ls = [r["manos_izq"] for r in rows if r["manos_izq"] != ""]
    rs = [r["manos_der"] for r in rows if r["manos_der"] != ""]
    ref = corpus_hand_rates(Path(bundle or "artifacts/data/v1/features.npz"))
    if ls and rs:
        print(f"  cuadros con mano izq.: {np.mean(ls) / 6:.2f}   der.: {np.mean(rs) / 6:.2f}" +
              (f"   | corpus train: izq. {ref[0]:.2f}, der. {ref[1]:.2f}" if ref else "   | (sin bundle para comparar)"))
    fr_all = [float(r["hombros_frac"]) for r in rows if r["hombros_frac"] != ""]
    if det is not None and fr_all:
        cref = corpus_shoulder_frac(det, Path(raw_dir) if raw_dir else None, corpus_sample)
        print(f"  hombros/ancho (mediana): {np.median(fr_all):.3f}" +
              (f"   | corpus ({corpus_sample} cuadros): {cref:.3f}" if cref else "   | (corpus no disponible para comparar)"))
    return 0
