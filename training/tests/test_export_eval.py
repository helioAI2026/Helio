import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

from eyestate.datasets import LiveClipAdapter, MRLAdapter
from eyestate.evaluate import evaluate
from eyestate.export import export
from eyestate.model import EyeNet


def _ckpt(tmp_path):
    torch.manual_seed(0)
    p = tmp_path / "best.pt"
    torch.save({"state_dict": EyeNet().state_dict(), "meta": {"split_seed": 0, "test_subjects": ["s0001"]}}, p)
    return p


def test_export_roundtrip(tmp_path):
    ckpt = _ckpt(tmp_path)
    out = export(ckpt, tmp_path / "eye_state.onnx")
    assert onnx.load(str(out)).opset_import[0].version == 13
    assert out.stat().st_size < 1_000_000

    model = EyeNet()
    model.load_state_dict(torch.load(ckpt, weights_only=False)["state_dict"])
    model.eval()
    x = torch.rand(4, 1, 64, 64)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(str(out)).run(None, {"input": x.numpy()})[0]
    assert got.shape == (4, 1)
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_evaluate_metrics_keys(tmp_path, fake_mrl):
    out = export(_ckpt(tmp_path), tmp_path / "m.onnx")
    m = evaluate(out, MRLAdapter(fake_mrl).samples())
    assert {"accuracy", "roc_auc", "confusion", "n"} <= m.keys()
    assert 0.0 <= m["accuracy"] <= 1.0
    assert sum(map(sum, m["confusion"])) == m["n"] == 200


def test_live_clip_adapter(tmp_path):
    for label in ("open", "closed"):
        (tmp_path / label).mkdir()
        cv2.imwrite(str(tmp_path / label / "00001.png"), np.zeros((64, 64), np.uint8))
    samples = {s.path.parent.name: s for s in LiveClipAdapter(tmp_path).samples()}
    assert samples["closed"].label == 1
    assert samples["open"].label == 0
    assert samples["open"].subject == "live"
