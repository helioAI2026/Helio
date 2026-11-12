import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from src.eye_state import pipeline as pl
from src.eye_state.classifier import EyeStateClassifier
from src.eye_state.geometry import Face


@pytest.fixture
def tiny_onnx(tmp_path):
    """input (N,1,64,64) -> logit = mean(pixels): bright crop => higher p_closed."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["n", 1, 64, 64])
    out = helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["n", 1])
    node = helper.make_node("ReduceMean", ["input"], ["logit"], axes=[2, 3], keepdims=0)
    model = helper.make_model(helper.make_graph([node], "tiny", [inp], [out]),
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
    p = tmp_path / "eye_state.onnx"
    onnx.save(model, str(p))
    return p


@pytest.fixture
def yunet_file(tmp_path):
    p = tmp_path / "yunet.onnx"
    p.write_bytes(b"fake")
    return p


def make_pipeline(monkeypatch, tiny_onnx, yunet_file, face):
    class FakeYuNet:
        def __init__(self, path):
            pass

        def detect(self, bgr):
            return face

    monkeypatch.setattr(pl, "YuNet", FakeYuNet)
    return pl.EyeStatePipeline(str(yunet_file), str(tiny_onnx))


def frame_with(right_val, left_val):
    """RGB float frame; right eye region around (200,240), left around (300,240)."""
    f = np.zeros((480, 640, 3), np.float32)
    f[200:280, 160:250] = right_val
    f[200:280, 250:340] = left_val
    return f


def face(right=(200, 240), left=(300, 240)):
    return Face(np.array([150, 150, 220, 220], np.float32), np.array(right, np.float32),
                np.array(left, np.float32), 0.95)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_classifier_predict_shape(tiny_onnx):
    p = EyeStateClassifier(str(tiny_onnx)).predict(np.full((2, 64, 64), 0.5, np.float32))
    assert p.shape == (2,)
    assert np.all((p > 0) & (p < 1))
    assert p[0] == pytest.approx(sigmoid(0.5), abs=1e-5)


def test_pipeline_mean_of_both_eyes(monkeypatch, tiny_onnx, yunet_file):
    r = make_pipeline(monkeypatch, tiny_onnx, yunet_file, face()).process(frame_with(1.0, 0.0))
    assert r.face_detected
    assert r.p_right == pytest.approx(sigmoid(1.0), abs=0.02)
    assert r.p_left == pytest.approx(sigmoid(0.0), abs=0.02)
    assert r.p_closed == pytest.approx((r.p_left + r.p_right) / 2)


def test_pipeline_one_eye_valid_uses_it(monkeypatch, tiny_onnx, yunet_file):
    f = face(right=(5, 240), left=(65, 240))  # right eye box leaves the frame
    r = make_pipeline(monkeypatch, tiny_onnx, yunet_file, f).process(frame_with(0.0, 0.0))
    assert r.boxes[0] is None and r.p_right is None
    assert r.p_closed == pytest.approx(r.p_left)


def test_pipeline_no_face(monkeypatch, tiny_onnx, yunet_file):
    r = make_pipeline(monkeypatch, tiny_onnx, yunet_file, None).process(frame_with(0.0, 0.0))
    assert r.p_closed is None
    assert r.face_detected is False


def test_missing_model_message(tiny_onnx, tmp_path):
    with pytest.raises(pl.ModelMissingError, match="download_models.py"):
        pl.EyeStatePipeline(str(tmp_path / "nope.onnx"), str(tiny_onnx))
    with pytest.raises(pl.ModelMissingError, match="eyestate.export"):
        pl.EyeStatePipeline(str(tiny_onnx), str(tmp_path / "nope.onnx"))


def test_missing_classifier_hint_says_train_then_export(tiny_onnx, tmp_path):
    with pytest.raises(pl.ModelMissingError) as e:
        pl.EyeStatePipeline(str(tiny_onnx), str(tmp_path / "nope.onnx"))
    msg = str(e.value)
    assert "cd training" in msg and "eyestate.train" in msg and "eyestate.export" in msg
