"""Exporta um checkpoint do EyeNet para ONNX (opset 13, compatível com TensorRT 8.0 / JetPack 4.6).

    uv run python -m eyestate.export --checkpoint checkpoints/best.pt
"""

import argparse
from pathlib import Path

import onnx
import torch

from eyestate.model import EyeNet

OPSET = 13


def export(checkpoint: Path, out_path: Path) -> Path:
    model = EyeNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, 1, 64, 64),),
        str(out_path),
        opset_version=OPSET,
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "n"}, "logit": {0: "n"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(out_path)))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("../model/eye_state.onnx"))
    a = ap.parse_args()
    p = export(a.checkpoint, a.out)
    print(f"{p} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
