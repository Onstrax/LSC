"""Funciones puras del diagnóstico de intentos reales (src/lsc_models/diagnose.py)."""

from __future__ import annotations

import json

import numpy as np

from src.lsc_models import diagnose as dg


def _x(seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (6, 49, 3)).astype(np.float32)
    z[..., 2] = (rng.random((6, 49)) > 0.3).astype(np.float32)  # máscara binaria
    return z.reshape(6, 147)


def test_mirror_is_an_involution():
    x = _x()
    again = dg.mirror_features(dg.mirror_features(x))
    np.testing.assert_allclose(again, x, atol=1e-6)


def test_mirror_negates_x_and_swaps_hands_and_shoulders():
    idx = dg.point_index()
    z = np.zeros((6, 49, 3), dtype=np.float32)
    z[:, idx["pose_11"]] = [0.5, -0.1, 1.0]    # hombro izq.
    z[:, idx["pose_12"]] = [-0.5, -0.1, 1.0]   # hombro der.
    z[:, idx["rh_0"]] = [-0.9, 0.7, 1.0]       # muñeca de la mano derecha
    m = dg.mirror_features(z.reshape(6, 147)).reshape(6, 49, 3)
    np.testing.assert_allclose(m[0, idx["pose_12"]], [-0.5, -0.1, 1.0])   # el izq. pasa a der. y se niega x
    np.testing.assert_allclose(m[0, idx["pose_11"]], [0.5, -0.1, 1.0])
    np.testing.assert_allclose(m[0, idx["lh_0"]], [0.9, 0.7, 1.0])        # la mano der. pasa a ser la izq., x negada
    assert m[0, idx["rh_0"], 2] == 0                                       # y la derecha queda ausente


def test_mirror_leaves_absent_points_at_zero():
    z = np.zeros((6, 49, 3), dtype=np.float32)
    m = dg.mirror_features(z.reshape(6, 147))
    assert not np.any(m)  # ausentes (m=0, x=y=0) siguen siendo 0...
    assert not np.signbit(m).any()  # ...y no se convierten en -0.0


def test_hand_presence_counts_frames():
    idx = dg.point_index()
    z = np.zeros((6, 49, 3), dtype=np.float32)
    for f in (0, 1, 2, 3):
        z[f, idx["lh_5"], 2] = 1.0
    z[5, idx["rh_20"], 2] = 1.0
    assert dg.hand_presence(z.reshape(6, 147)) == (4, 1)


def test_parse_expected():
    assert dg.parse_expected("Hola*3,Buenos,Días*2") == ["Hola"] * 3 + ["Buenos"] + ["Días"] * 2
    assert dg.parse_expected(None) == [] and dg.parse_expected("  , ") == []


def test_load_attempts_is_chronological_and_skips_incomplete(tmp_path):
    for name in ("20261120_100000_000_b", "20261120_090000_000_a", "sin_meta"):
        (tmp_path / name).mkdir()
    for name in ("20261120_100000_000_b", "20261120_090000_000_a"):
        (tmp_path / name / "meta.json").write_text(json.dumps({"quality_pass": True}), encoding="utf-8")
    np.save(tmp_path / "20261120_090000_000_a" / "x.npy", _x())
    got = dg.load_attempts(tmp_path)
    assert [a["dir"].name for a in got] == ["20261120_090000_000_a", "20261120_100000_000_b"]
    assert got[0]["x"].shape == (6, 147) and got[1]["x"] is None


def test_run_diagnose_reports_missing_dump(tmp_path, capsys):
    assert dg.run_diagnose(str(tmp_path / "no_existe"), "artifacts", None) == 3
    assert "debug-dump" in capsys.readouterr().err
