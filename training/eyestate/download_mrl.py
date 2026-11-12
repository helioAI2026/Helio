"""Baixa e descompacta o MRL Eye Dataset em data/mrl/.

    uv run python -m eyestate.download_mrl
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()

    args.data.mkdir(parents=True, exist_ok=True)
    zip_path = args.data / "mrlEyes_2018_01.zip"
    out = args.data / "mrl"

    if not zip_path.exists():
        print(f"Baixando {URL} ...")
        try:
            urllib.request.urlretrieve(URL, zip_path)
        except Exception as e:  # noqa: BLE001
            zip_path.unlink(missing_ok=True)
            sys.exit(
                f"Falha ao baixar ({e}).\nBaixe manualmente {URL} e salve em {zip_path}, depois rode de novo."
            )

    print(f"Descompactando em {out} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    n = sum(1 for _ in out.rglob("*.png"))
    print(f"Pronto: {n} imagens.")


if __name__ == "__main__":
    main()
