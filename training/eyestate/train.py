"""Treina o EyeNet.

    uv run python -m eyestate.train --data data/mrl --out checkpoints
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from eyestate.datasets import DatasetAdapter, EyeDataset, MRLAdapter, subject_split
from eyestate.model import EyeNet

SPLIT_SEED = 0


def pick_device(device: str | None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = (model(x.to(device)).squeeze(1) > 0).float().cpu()
        correct += int((pred == y).sum())
        total += len(y)
    return correct / max(total, 1)


def train(
    data_root: Path,
    out_dir: Path,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | None = None,
    adapter: DatasetAdapter | None = None,
    workers: int = 0,
) -> Path:
    device = pick_device(device)
    samples = (adapter or MRLAdapter(data_root)).samples()
    if not samples:
        raise SystemExit(f"Nenhuma imagem encontrada em {data_root}")
    tr, va, te = subject_split(samples, seed=SPLIT_SEED)
    print(f"device={device} train={len(tr)} val={len(va)} test={len(te)} "
          f"subjects test={sorted({s.subject for s in te})}")

    dl_kw = dict(batch_size=batch_size, num_workers=workers, persistent_workers=workers > 0)
    train_dl = DataLoader(EyeDataset(tr, train=True), shuffle=True, **dl_kw)
    val_dl = DataLoader(EyeDataset(va, train=False), **dl_kw)

    n_closed = sum(s.label for s in tr)
    pos_weight = torch.tensor([(len(tr) - n_closed) / max(n_closed, 1)], device=device)

    model = EyeNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_dl))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_acc = -1.0
    meta = {"split_seed": SPLIT_SEED, "test_subjects": sorted({s.subject for s in te})}

    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(y)
        val_acc = accuracy(model, val_dl, device)
        print(f"epoch {epoch:2d} loss={running / len(tr):.4f} val_acc={val_acc:.4f} ({time.time() - t0:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "meta": {**meta, "val_acc": val_acc}}, best_path)

    print(f"melhor val_acc={best_acc:.4f} -> {best_path}")
    return best_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/mrl"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    train(a.data, a.out, a.epochs, a.batch_size, a.lr, a.device, workers=a.workers)


if __name__ == "__main__":
    main()
