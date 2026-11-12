"""Baixa o detector de rosto YuNet (OpenCV Zoo) para model/.

    uv run python scripts/download_models.py
"""

import subprocess
import sys
from pathlib import Path

URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
OUT = Path(__file__).resolve().parent.parent / "model" / "face_detection_yunet_2023mar.onnx"


def main() -> None:
    if OUT.exists():
        print(f"Já existe: {OUT}")
        return
    OUT.parent.mkdir(exist_ok=True)
    print(f"Baixando {URL} ...")
    # curl usa o repositório de certificados do macOS (funciona atrás de proxies com inspeção TLS)
    r = subprocess.run(["curl", "-fsSL", "--max-time", "120", "-o", str(OUT), URL])
    if r.returncode != 0:
        OUT.unlink(missing_ok=True)
        sys.exit(f"Falha ao baixar (curl saiu com {r.returncode}). Baixe manualmente {URL} e salve em {OUT}.")
    if OUT.read_bytes()[:20].lstrip().lower().startswith((b"<!doctype", b"<html")):
        OUT.unlink()
        sys.exit(f"O download retornou uma página HTML (rede bloqueando?). Baixe manualmente {URL} e salve em {OUT}.")
    print(f"Pronto: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
