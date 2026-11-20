"""Volcado local de diagnóstico (api/debug_dump.py): apagado por defecto, no altera la respuesta."""

from __future__ import annotations

import json

import numpy as np

from api.debug_dump import ENV_VAR, DebugDump, presence_per_frame
from api.worker import JobResult
from tests.app.conftest import build_multipart, make_jpeg, valid_metadata


def _x_with(left_hand_frames=(), right_hand_frames=(), shoulders=True):
    z = np.zeros((6, 49, 3), dtype=np.float32)
    if shoulders:
        z[:, 1, 2] = 1.0
        z[:, 2, 2] = 1.0
    for f in left_hand_frames:
        z[f, 7:28, 2] = 1.0
    for f in right_hand_frames:
        z[f, 28:49, 2] = 1.0
    return z.reshape(6, 147)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert DebugDump.from_env() is None
    monkeypatch.setenv(ENV_VAR, "   ")
    assert DebugDump.from_env() is None


def test_enabled_by_env_creates_the_folder(monkeypatch, tmp_path):
    target = tmp_path / "dump"
    monkeypatch.setenv(ENV_VAR, str(target))
    dd = DebugDump.from_env()
    assert dd is not None and target.is_dir()


def test_presence_per_frame():
    p = presence_per_frame(_x_with(left_hand_frames=[0, 1, 2], right_hand_frames=[1, 5], shoulders=True))
    assert p["left_hand"] == [1, 1, 1, 0, 0, 0]
    assert p["right_hand"] == [0, 1, 0, 0, 0, 1]
    assert p["left_shoulder"] == [1] * 6 and p["right_shoulder"] == [1] * 6


def test_save_writes_frames_x_and_meta(tmp_path):
    probs = np.zeros(10, dtype=np.float32)
    probs[3], probs[7] = 0.7, 0.3
    result = JobResult(quality_pass=True, accepted=False, label_id=None, confidence=None, reason_codes=["LOW_CONFIDENCE"],
                       model_version="m-1", timings_ms={"inference": 1.0}, x=_x_with([0], [0, 1]), probs=probs)
    labels = [f"L{i}" for i in range(10)]
    out = DebugDump(tmp_path).save(raw_frames=[b"jpg%d" % i for i in range(6)], meta={"client_request_id": "abc-123", "width": 640, "height": 480},
                                   result=result, labels=labels)
    assert out is not None and out.parent == tmp_path
    assert [p.name for p in sorted(out.glob("frame_*.jpg"))] == [f"frame_{i}.jpg" for i in range(6)]
    assert np.load(out / "x.npy").shape == (6, 147)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["reason_codes"] == ["LOW_CONFIDENCE"] and meta["quality_pass"] is True
    assert meta["top3"][0] == {"label_id": 3, "label": "L3", "prob": meta["top3"][0]["prob"]}
    assert abs(meta["top3"][0]["prob"] - 0.7) < 1e-6 and len(meta["probs"]) == 10
    assert meta["presence"]["right_hand"] == [1, 1, 0, 0, 0, 0]


def test_quality_failure_still_saves_x_without_probs(tmp_path):
    result = JobResult(quality_pass=False, accepted=False, label_id=None, confidence=None,
                       reason_codes=["INSUFFICIENT_HANDS"], x=_x_with(), probs=None)
    out = DebugDump(tmp_path).save(raw_frames=[b"a"] * 6, meta={}, result=result, labels=[])
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert "probs" not in meta and meta["quality_pass"] is False


def test_save_never_raises(tmp_path):
    # carpeta imposible (un archivo en lugar de directorio) y resultado incompleto
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert DebugDump(blocker).save(raw_frames=[b"a"], meta={}, result=object(), labels=[]) is None


def test_http_response_is_identical_with_and_without_dump(client, tmp_path):
    frames = [make_jpeg(color=10 + i) for i in range(6)]
    meta = valid_metadata()

    def post():
        files, data = build_multipart(frames, meta)
        return client.post("/api/v1/predict", files=files, data=data)

    state = client.app.state.lsc
    assert state.debug_dump is None  # apagado por defecto
    baseline = post().json()

    state.debug_dump = DebugDump(tmp_path)
    with_dump = post().json()
    for k in ("status", "label_id", "display_text", "confidence", "reason_codes", "model_version"):
        assert baseline[k] == with_dump[k]
    saved = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(saved) == 1 and len(list(saved[0].glob("frame_*.jpg"))) == 6
    assert (saved[0] / "meta.json").is_file()
