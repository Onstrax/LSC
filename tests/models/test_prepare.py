import numpy as np
import pytest

from src.lsc_models import prepare
from src.lsc_models.config import ConfigError
from src.lsc_models.data import Bundle
from src.lsc_models.experiments import run_all


def test_missing_prerequisites_are_listed_not_invented(tmp_path):
    problems = prepare.check_prerequisites(tmp_path / "no_corpus", tmp_path / "no_models")
    text = "\n".join(problems)
    assert "9ssyn8tff5" in text and "pose_landmarker.task" in text and "hand_landmarker.task" in text


def test_real_path_refuses_without_prereqs(tmp_path):
    with pytest.raises(prepare.PrepareError, match="REAL todavía"):
        prepare.prepare_real(tmp_path / "x", tmp_path / "y", tmp_path / "out")
    assert not (tmp_path / "out").exists()  # no fabricó nada


def test_main_prepare_exit_code_3_when_missing(tmp_path, capsys):
    rc = prepare.main_prepare(str(tmp_path / "x"), str(tmp_path / "y"), str(tmp_path / "o"), synthetic=False)
    assert rc == 3 and "verify" in capsys.readouterr().err


def test_wrong_task_hash_detected(tmp_path):
    raw = tmp_path / "raw" / "Per01"
    raw.mkdir(parents=True)
    models = tmp_path / "models"
    models.mkdir()
    for f in prepare.TASK_FILES.values():
        (models / f).write_bytes(b"not the real task")
    warns = []
    problems = prepare.check_prerequisites(tmp_path / "raw", models, warns)
    # pose: hash válido y distinto -> BLOQUEA; hands: hash mal formado en el contrato -> solo aviso
    assert len(problems) == 1 and "pose_landmarker.task" in problems[0] and "SHA-256" in problems[0]
    assert any("hands" in w for w in warns)


def test_synthetic_goes_to_separate_folder_and_is_flagged(tmp_path):
    p = prepare.prepare_synthetic(tmp_path / "syn")
    assert Bundle(p).synthetic and (tmp_path / "syn" / "README_SYNTHETIC.txt").is_file()


def test_synthetic_never_written_to_v1():
    with pytest.raises(prepare.PrepareError):
        prepare.prepare_synthetic(prepare.DEFAULT_OUT)


def test_real_pipeline_runs_steps_in_order_and_stops_on_failure(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "Per01"
    raw.mkdir(parents=True)
    monkeypatch.setattr(prepare, "check_prerequisites", lambda *_: [])
    monkeypatch.setattr(prepare, "INTERIM", tmp_path / "interim")
    calls = []

    def fake_main(argv):
        calls.append(argv[0])
        return 3 if argv[0] == "features" else 0

    import src.lsc_data.cli as cli

    monkeypatch.setattr(cli, "main", fake_main)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(prepare.PrepareError, match="código 3"):
        prepare.prepare_real(tmp_path / "raw", tmp_path / "m", tmp_path / "out", download=False)
    assert calls == ["inventory", "manifest", "split", "features"]  # se detuvo; no llegó a validate


def test_fast_refused_with_real_data(tmp_path, synthetic_bundle_path):
    d = dict(np.load(synthetic_bundle_path, allow_pickle=False))
    d.pop("synthetic_contract_fixture")  # simula un bundle "real"
    np.savez(tmp_path / "real_like.npz", **d)
    with pytest.raises(ConfigError, match="solo se permite con datos sintéticos"):
        run_all("configs/train/experiments.yaml", tmp_path / "o", bundle_path=tmp_path / "real_like.npz", fast=True)


def test_malformed_contract_hash_warns_but_does_not_block(tmp_path):
    spec = {"model_asset_hashes": {"pose": "a" * 64, "hands": "b" * 65}}
    w = prepare.contract_warnings(spec)
    assert len(w) == 1 and "hands" in w[0] and "65" in w[0]


def test_real_contract_currently_has_the_65_char_defect_flagged():
    # Documenta el hallazgo sobre contracts/feature_spec.json de Ortiz (no se corrige aquí).
    assert any("hands" in w for w in prepare.contract_warnings())


def _spec_for(pose: bytes, hands: bytes, malformed_hands: bool = False):
    import hashlib

    h = hashlib.sha256(hands).hexdigest()
    return {"model_asset_hashes": {"pose": hashlib.sha256(pose).hexdigest(), "hands": h + "1" if malformed_hands else h}}


def test_download_keeps_only_the_variant_that_matches(tmp_path):
    good, bad = b"pose-full-bytes", b"pose-heavy-bytes"
    spec = _spec_for(good, b"hands-bytes")
    blobs = {u: (bad if "heavy" in u else good if "full" in u else b"pose-lite") for u in prepare.POSE_CANDIDATES}
    blobs[prepare.HANDS_CANDIDATES[0]] = b"hands-bytes"
    st = prepare.download_models(tmp_path, fetch=lambda u: blobs[u], spec=spec)
    assert (tmp_path / "pose_landmarker.task").read_bytes() == good  # no se quedó con heavy
    assert "verificado" in st["pose"] and "verificado" in st["hands"]


def test_download_accepts_malformed_65_char_contract_hash(tmp_path):
    spec = _spec_for(b"p", b"h", malformed_hands=True)
    blobs = {u: b"p" for u in prepare.POSE_CANDIDATES} | {prepare.HANDS_CANDIDATES[0]: b"h"}
    prepare.download_models(tmp_path, fetch=lambda u: blobs[u], spec=spec)
    assert (tmp_path / "hand_landmarker.task").is_file()


def test_download_refuses_unverified_files_by_default(tmp_path):
    spec = _spec_for(b"expected-pose", b"expected-hands")
    with pytest.raises(prepare.PrepareError, match="ninguna variante coincide"):
        prepare.download_models(tmp_path, fetch=lambda u: b"something else", spec=spec)
    assert not list(tmp_path.glob("*.task"))  # no dejó archivos sin verificar


def test_download_accept_mismatch_is_explicit_and_loud(tmp_path):
    spec = _spec_for(b"expected-pose", b"expected-hands")
    st = prepare.download_models(tmp_path, fetch=lambda u: b"something else", spec=spec, accept_mismatch=True)
    assert "SIN VERIFICAR" in st["pose"] and (tmp_path / "pose_landmarker.task").is_file()


def test_download_offline_reports_clear_error(tmp_path):
    def offline(url):
        raise OSError("Could not resolve host")

    with pytest.raises(prepare.PrepareError, match="Sin internet"):
        prepare.download_models(tmp_path, fetch=offline, spec=_spec_for(b"a", b"b"))


def _mk_corpus(root):
    for sub in ("LSC70W", "LSC70ANH", "LSC70AN"):
        (root / "LSC70" / "LSC70" / sub).mkdir(parents=True)
    for per in ("Per01", "Per02"):
        (root / "LSC70" / "LSC70" / "LSC70W" / per / "HOLA").mkdir(parents=True)
    (root / "LSC70" / "LSC70" / "LSC70ANH" / "Per01").mkdir()  # hermanas con Per##, NO deben usarse
    return root / "LSC70" / "LSC70" / "LSC70W"


def test_autodetects_lsc70w_without_outer_folder(tmp_path):
    w = _mk_corpus(tmp_path)
    assert prepare.resolve_raw_dir(None, search_from=tmp_path).resolve() == w.resolve()


def test_descends_from_parent_folder(tmp_path):
    w = _mk_corpus(tmp_path)
    assert prepare.resolve_raw_dir(tmp_path / "LSC70").resolve() == w.resolve()
    assert prepare.resolve_raw_dir(w) == w  # ya es LSC70W


def test_sibling_subsets_are_ignored(tmp_path):
    _mk_corpus(tmp_path)
    found = prepare.find_lsc70w(tmp_path)
    assert len(found) == 1 and found[0].name == "LSC70W"


def test_ambiguous_corpus_asks_user(tmp_path):
    _mk_corpus(tmp_path / "a")
    _mk_corpus(tmp_path / "b")
    with pytest.raises(prepare.AmbiguousRawDir, match="--raw-dir"):
        prepare.resolve_raw_dir(None, search_from=tmp_path)


def test_explicit_missing_path_is_not_replaced(tmp_path):
    p = tmp_path / "nope"
    assert prepare.resolve_raw_dir(p) == p


def test_download_accepts_by_prefix_when_contract_hash_is_garbled(tmp_path):
    import hashlib
    import json

    real = hashlib.sha256(b"hands-bytes").hexdigest()
    garbled = real[:12] + "0" * 52 + "1"  # 65 caracteres, mismo comienzo, resto distinto
    spec = {"model_asset_hashes": {"pose": hashlib.sha256(b"p").hexdigest(), "hands": garbled}}
    blobs = {u: b"p" for u in prepare.POSE_CANDIDATES} | {prepare.HANDS_CANDIDATES[0]: b"hands-bytes"}
    st = prepare.download_models(tmp_path, fetch=lambda u: blobs[u], spec=spec)
    assert "POR PREFIJO" in st["hands"] and real in st["hands"]
    obs = json.loads((tmp_path / "hashes_observed.json").read_text())
    assert obs["hands"]["sha256"] == real and obs["hands"]["match"] == "prefijo"
    assert obs["pose"]["match"] == "exacto"


def test_download_rejects_when_even_the_prefix_differs(tmp_path):
    import hashlib

    garbled = "a" * 65
    spec = {"model_asset_hashes": {"pose": hashlib.sha256(b"p").hexdigest(), "hands": garbled}}
    blobs = {u: b"p" for u in prepare.POSE_CANDIDATES} | {prepare.HANDS_CANDIDATES[0]: b"hands-bytes"}
    with pytest.raises(prepare.PrepareError, match="ninguna variante coincide"):
        prepare.download_models(tmp_path, fetch=lambda u: blobs[u], spec=spec)
    assert not (tmp_path / "hand_landmarker.task").exists()
