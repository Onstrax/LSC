import numpy as np
import pytest

from src.lsc_models.data import Bundle, BundleError, TestAccessError


def test_bundle_marked_synthetic(bundle):
    assert bundle.synthetic is True


def test_shapes_and_dtypes(bundle):
    tr = bundle.load_split("train")
    assert tr.X.dtype == np.float32 and tr.X.shape[1:] == (6, 147)
    assert tr.y.dtype == np.int64


def test_processable_filters_quality(bundle):
    tr = bundle.load_split("train")
    pr = tr.processable()
    assert len(pr) == int(tr.quality_pass.sum()) < len(tr)


def test_test_split_blocked_by_default(bundle):
    with pytest.raises(TestAccessError):
        bundle.load_split("test")


def test_test_split_blocked_without_freeze_file(bundle):
    with pytest.raises(TestAccessError):
        bundle.load_split("test", allow_test=True)


def test_test_split_allowed_only_after_freeze(bundle):
    bundle.freeze_file.write_text("{}")
    assert len(bundle.load_split("test", allow_test=True)) > 0


def test_summary_does_not_expose_test_content(bundle):
    s = bundle.summary()
    assert "n_test" in s and "X" not in s


def test_no_signer_in_two_splits(bundle):
    seen = {}
    for sp in ("train", "val"):
        for s in bundle.load_split(sp).signer_id:
            assert seen.setdefault(s, sp) == sp


def test_leak_detected(tmp_path, synthetic_bundle_path):
    d = dict(np.load(synthetic_bundle_path, allow_pickle=False))
    d["split"] = d["split"].copy()
    s0 = d["signer_id"][0]
    idx = np.where(d["signer_id"] == s0)[0]
    d["split"][idx[0]] = "val" if d["split"][idx[0]] != "val" else "train"
    np.savez(tmp_path / "leak.npz", **d)
    with pytest.raises(BundleError, match="fuga"):
        Bundle(tmp_path / "leak.npz")


def test_missing_bundle_does_not_invent_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="No se inventan datos"):
        Bundle(tmp_path / "nope.npz")
