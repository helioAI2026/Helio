"""Grava um clipe de avaliação ao vivo com a webcam (só para avaliar, nunca para treinar).

    uv run python -m eyestate.record_eval --out data/live

Aperte "o" e fique com os olhos ABERTOS, ou "c" e fique com os olhos FECHADOS;
espaço pausa a gravação; "q" sai.
Os recortes dos dois olhos são salvos em data/live/{open,closed}/NNNNN.png, usando o
mesmo recorte do app (YuNet + geometry), para medir o modelo nas mesmas condições.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

EDGE_ROOT = Path(__file__).resolve().parents[2]


def save_crop(crop: np.ndarray, out: Path, label: str) -> Path:
    d = Path(out) / label
    d.mkdir(parents=True, exist_ok=True)
    idx = len(list(d.glob("*.png")))
    path = d / f"{idx:05d}.png"
    cv2.imwrite(str(path), (np.clip(crop, 0, 1) * 255).astype(np.uint8))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/live"))
    ap.add_argument("--camera", type=int, default=int(os.environ.get("HELIO_CAMERA", "0")))
    a = ap.parse_args()

    sys.path.insert(0, str(EDGE_ROOT))
    import config
    from src.eye_state.geometry import crop_eye, eye_boxes
    from src.eye_state.yunet import YuNet

    det = YuNet(str(EDGE_ROOT / config.YUNET_MODEL))
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit("Câmera não disponível (verifique a permissão de câmera do terminal).")
    counts = {"open": 0, "closed": 0}
    held = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        face = det.detect(bgr)
        boxes = eye_boxes(face, bgr.shape) if face is not None else (None, None)
        if held:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for b in boxes:
                if b is not None:
                    save_crop(crop_eye(gray, b), a.out, held)
                    counts[held] += 1
        for b in boxes:
            if b is not None:
                cv2.rectangle(bgr, b[:2], b[2:], (0, 0, 255) if held == "closed" else (0, 200, 0), 1)
        cv2.putText(bgr, f"[{held or 'pausado'}] o: aberto ({counts['open']})  c: fechado ({counts['closed']})  espaco: pausa  q: sair",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Helio - gravar avaliacao", bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            held = None
        elif key in (ord("o"), ord("c")):
            held = "open" if key == ord("o") else "closed"
    cap.release()
    cv2.destroyAllWindows()
    print(f"Salvo em {a.out}: {counts}")


if __name__ == "__main__":
    main()
