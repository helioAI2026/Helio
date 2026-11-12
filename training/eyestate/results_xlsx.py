"""Exporta os resultados de treino/avaliação para uma planilha Excel.

    uv run --with openpyxl python -m eyestate.results_xlsx --out results/eye_state_results.xlsx

Lê o log de treino, roda o modelo ONNX nos sujeitos de teste do MRL e no clipe da webcam,
e grava contagens brutas; acurácias e totais são fórmulas do Excel.
"""

import argparse
import re
from datetime import date
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from sklearn.metrics import roc_auc_score

from eyestate.datasets import LiveClipAdapter, MRLAdapter
from eyestate.evaluate import predict
from eyestate.model import EyeNet

FONT = "Arial"
BLUE = "0000FF"
HEADER_FILL = PatternFill("solid", start_color="1F3864")
TITLE = Font(name=FONT, size=14, bold=True)
HEAD = Font(name=FONT, bold=True, color="FFFFFF")
BODY = Font(name=FONT)
INPUT = Font(name=FONT, color=BLUE)
NOTE = Font(name=FONT, italic=True, color="595959", size=9)
THIN = Border(bottom=Side(style="thin", color="BFBFBF"))
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_log(path: Path) -> tuple[dict, list[tuple[int, float, float, int]]]:
    text = path.read_text()
    head = re.search(r"device=(\S+) train=(\d+) val=(\d+) test=(\d+) subjects test=\[(.*?)\]", text)
    info = {"device": head[1], "train": int(head[2]), "val": int(head[3]), "test": int(head[4]),
            "test_subjects": head[5].replace("'", "")}
    epochs = [(int(e), float(l), float(a), int(s)) for e, l, a, s in
              re.findall(r"epoch\s+(\d+) loss=([\d.]+) val_acc=([\d.]+) \((\d+)s\)", text)]
    return info, epochs


def confusion(p: np.ndarray, y: np.ndarray, thr: float = 0.5) -> tuple[int, int, int, int]:
    pred = p > thr
    return (int(((~pred) & (y == 0)).sum()), int((pred & (y == 0)).sum()),
            int(((~pred) & (y == 1)).sum()), int((pred & (y == 1)).sum()))


def header(ws, row: int, labels: list[str]) -> None:
    for c, label in enumerate(labels, 1):
        cell = ws.cell(row, c, label)
        cell.font, cell.fill = HEAD, HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def put(ws, row: int, col: int, value, fmt: str | None = None, font: Font = BODY):
    cell = ws.cell(row, col, value)
    cell.font = font
    if fmt:
        cell.number_format = fmt
    return cell


def widths(ws, cols: dict[str, float]) -> None:
    for col, w in cols.items():
        ws.column_dimensions[col].width = w


def confusion_block(ws, title: str, counts: tuple[int, int, int, int], auc: float, note: str) -> None:
    """Rows 3-6 hold the 2x2 counts (inputs); rows 9-14 hold formula metrics."""
    ws["A1"], ws["A1"].font = title, TITLE
    header(ws, 3, ["Actual \\ Predicted", "Predicted open", "Predicted closed", "Total"])
    tn, fp, fn, tp = counts
    for r, (label, a, b) in enumerate([("Actual open", tn, fp), ("Actual closed", fn, tp)], 4):
        put(ws, r, 1, label, font=Font(name=FONT, bold=True))
        put(ws, r, 2, a, "#,##0", INPUT)
        put(ws, r, 3, b, "#,##0", INPUT)
        put(ws, r, 4, f"=SUM(B{r}:C{r})", "#,##0")
    put(ws, 6, 1, "Total", font=Font(name=FONT, bold=True))
    for c in "BCD":
        put(ws, 6, "BCD".index(c) + 2, f"=SUM({c}4:{c}5)", "#,##0")

    header(ws, 8, ["Metric", "Value", "How it is computed"])
    rows = [
        ("Accuracy", "=(B4+C5)/D6", "correct / all images"),
        ("Open eyes read correctly (specificity)", "=B4/D4", "open→open / all open"),
        ("Closed eyes caught (recall)", "=C5/D5", "closed→closed / all closed"),
        ("Precision of 'closed'", "=IF(C6=0,0,C5/C6)", "true closed / everything predicted closed"),
        ("Open misread as closed (false alarms)", "=C4/D4", "pushes PERCLOS up"),
        ("Closed misread as open (missed)", "=B5/D5", "pushes PERCLOS down"),
    ]
    for r, (label, formula, how) in enumerate(rows, 9):
        put(ws, r, 1, label)
        put(ws, r, 2, formula, "0.0%")
        put(ws, r, 3, how, font=NOTE)
    put(ws, 15, 1, "ROC-AUC")
    put(ws, 15, 2, round(auc, 4), "0.000", INPUT).comment = Comment(
        "Computed in Python with sklearn.metrics.roc_auc_score on the model's probabilities "
        "(not derivable from the 2×2 counts).", "Helio")
    put(ws, 15, 3, "area under the ROC curve (1.0 = perfect)", font=NOTE)
    put(ws, 17, 1, note, font=NOTE)
    put(ws, 18, 1, "Blue = values measured by running the model; black = Excel formulas. Threshold 0.5.",
        font=NOTE)
    widths(ws, {"A": 40, "B": 16, "C": 44, "D": 12})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=Path("../model/eye_state.onnx"))
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    ap.add_argument("--log", type=Path, default=Path("checkpoints_train.log"))
    ap.add_argument("--data", type=Path, default=Path("data/mrl"))
    ap.add_argument("--live", type=Path, default=Path("data/live"))
    ap.add_argument("--out", type=Path, default=Path("results/eye_state_results.xlsx"))
    a = ap.parse_args()

    info, epochs = parse_log(a.log)
    meta = torch.load(a.checkpoint, weights_only=False)["meta"]
    session = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    test = [s for s in MRLAdapter(a.data).samples() if s.subject in set(meta["test_subjects"])]
    live = LiveClipAdapter(a.live).samples()
    p_mrl, y_mrl = predict(session, test), np.array([s.label for s in test])
    p_live, y_live = predict(session, live), np.array([s.label for s in live])
    subj_mrl = np.array([s.subject for s in test])

    wb = Workbook()

    # ---- Training ----
    tr = wb.active
    tr.title = "Training"
    tr["A1"], tr["A1"].font = "Training run — EyeNet on MRL Eye Dataset", TITLE
    header(tr, 3, ["Epoch", "Train loss", "Validation accuracy", "Epoch time (s)"])
    for i, (e, loss, acc, sec) in enumerate(epochs, 4):
        put(tr, i, 1, e, "0", INPUT)
        put(tr, i, 2, loss, "0.0000", INPUT)
        put(tr, i, 3, acc, "0.00%", INPUT)
        put(tr, i, 4, sec, "0", INPUT)
    last = 3 + len(epochs)
    s = last + 2
    put(tr, s, 1, "Best validation accuracy", font=Font(name=FONT, bold=True))
    put(tr, s, 3, f"=MAX(C4:C{last})", "0.00%")
    put(tr, s + 1, 1, "Best epoch (checkpoint kept)", font=Font(name=FONT, bold=True))
    put(tr, s + 1, 3, f"=INDEX(A4:A{last},MATCH(C{s},C4:C{last},0))", "0")
    put(tr, s + 2, 1, "Total training time (min)", font=Font(name=FONT, bold=True))
    put(tr, s + 2, 3, f"=SUM(D4:D{last})/60", "0.0")
    put(tr, s + 4, 1, f"Source: {a.log.name} (output of eyestate.train). Blue = logged values.", font=NOTE)
    widths(tr, {"A": 30, "B": 14, "C": 20, "D": 16})

    # ---- MRL test / Webcam ----
    mrl = wb.create_sheet("MRL Test")
    confusion_block(mrl, "MRL test set — 6 people never seen in training", confusion(p_mrl, y_mrl),
                    roc_auc_score(y_mrl, p_mrl),
                    f"Test subjects: {info['test_subjects']}. Infrared eye crops from the MRL Eye Dataset.")
    r0 = 21
    mrl.cell(r0 - 1, 1, "Accuracy per test person").font = Font(name=FONT, bold=True)
    header(mrl, r0, ["Subject", "Images", "Correct", "Accuracy"])
    for i, sid in enumerate(sorted(set(subj_mrl)), r0 + 1):
        m = subj_mrl == sid
        put(mrl, i, 1, sid)
        put(mrl, i, 2, int(m.sum()), "#,##0", INPUT)
        put(mrl, i, 3, int(((p_mrl[m] > 0.5) == y_mrl[m]).sum()), "#,##0", INPUT)
        put(mrl, i, 4, f"=C{i}/B{i}", "0.0%")
    end = r0 + len(set(subj_mrl))
    put(mrl, end + 1, 1, "All", font=Font(name=FONT, bold=True))
    put(mrl, end + 1, 2, f"=SUM(B{r0 + 1}:B{end})", "#,##0")
    put(mrl, end + 1, 3, f"=SUM(C{r0 + 1}:C{end})", "#,##0")
    put(mrl, end + 1, 4, f"=C{end + 1}/B{end + 1}", "0.0%")

    cam = wb.create_sheet("Webcam")
    confusion_block(cam, "Webcam check — MacBook camera, 1 person, evaluation only",
                    confusion(p_live, y_live), roc_auc_score(y_live, p_live),
                    "Eye crops recorded with eyestate.record_eval (YuNet + same cropping as the app). "
                    "Never used for training.")

    # ---- Threshold sweep ----
    sw = wb.create_sheet("Threshold Sweep")
    sw["A1"], sw["A1"].font = "Effect of the 'closed' threshold (app uses 0.5)", TITLE
    header(sw, 3, ["Threshold", "MRL images", "MRL correct", "MRL accuracy",
                   "Webcam images", "Webcam correct", "Webcam accuracy",
                   "Webcam closed eyes", "Webcam closed caught", "Webcam closed recall"])
    for i, t in enumerate(THRESHOLDS, 4):
        put(sw, i, 1, t, "0.0", INPUT)
        put(sw, i, 2, len(y_mrl), "#,##0", INPUT)
        put(sw, i, 3, int(((p_mrl > t) == y_mrl).sum()), "#,##0", INPUT)
        put(sw, i, 4, f"=C{i}/B{i}", "0.0%")
        put(sw, i, 5, len(y_live), "#,##0", INPUT)
        put(sw, i, 6, int(((p_live > t) == y_live).sum()), "#,##0", INPUT)
        put(sw, i, 7, f"=F{i}/E{i}", "0.0%")
        put(sw, i, 8, int((y_live == 1).sum()), "#,##0", INPUT)
        put(sw, i, 9, int(((p_live > t) & (y_live == 1)).sum()), "#,##0", INPUT)
        put(sw, i, 10, f"=I{i}/H{i}", "0.0%")
        if t == 0.5:
            for c in range(1, 11):
                sw.cell(i, c).fill = PatternFill("solid", start_color="FFF2CC")
    put(sw, 5 + len(THRESHOLDS), 1, "Highlighted row = threshold in config.py (P_CLOSED_THRESHOLD). "
        "Lower threshold catches more closed eyes but adds false alarms on open eyes.", font=NOTE)
    widths(sw, {get_column_letter(c): 14 for c in range(1, 11)})

    # ---- Setup ----
    st = wb.create_sheet("Setup")
    st["A1"], st["A1"].font = "Model and data setup", TITLE
    params = sum(p.numel() for p in EyeNet().parameters())
    model = onnx.load(str(a.onnx))
    rows = [
        ("Dataset", "MRL Eye Dataset (mrlEyes_2018_01), infrared eye crops, research use"),
        ("Split", "by person (no person in more than one split), seed 0"),
        ("Training images", info["train"]),
        ("Validation images", info["val"]),
        ("Test images", info["test"]),
        ("Test people", info["test_subjects"]),
        ("Model", "EyeNet: 4 conv blocks (16/32/64/96) + global avg pool + linear"),
        ("Parameters", params),
        ("Input", "grayscale 64×64, values 0–1"),
        ("ONNX file size (KB)", round(a.onnx.stat().st_size / 1024)),
        ("ONNX opset / IR version", f"{model.opset_import[0].version} / {model.ir_version}"),
        ("Training device", f"{info['device']} (Apple M4 GPU)"),
        ("Epochs / batch size / learning rate", "15 / 256 / 0.001 (AdamW, cosine schedule)"),
        ("Webcam FPS (app, eyestate mode)", "30 (camera limit) — reported by user"),
        ("Pipeline FPS without camera", "~120 at 640×480 on CPU — measured in session"),
        ("Report generated", date.today().isoformat()),
    ]
    for i, (k, v) in enumerate(rows, 3):
        put(st, i, 1, k, font=Font(name=FONT, bold=True))
        put(st, i, 2, v, "#,##0" if isinstance(v, int) else None, INPUT if isinstance(v, int) else BODY)
    widths(st, {"A": 36, "B": 70})

    # ---- Summary (first tab) ----
    sm = wb.create_sheet("Summary", 0)
    sm["A1"], sm["A1"].font = "Helio eye-state model — results", TITLE
    put(sm, 2, 1, "Open/closed eye classifier for drowsiness detection, tested on a MacBook (M4).", font=NOTE)
    header(sm, 4, ["Goal", "Target", "Result", "Status"])
    goals = [
        ("Accuracy on unseen people (MRL test)", 0.95, "='MRL Test'!B9", "0.0%", ">="),
        ("Accuracy on the webcam", 0.90, "=Webcam!B9", "0.0%", ">="),
        ("Model size (KB)", 1024, "=Setup!B12", "#,##0", "<="),
        ("Frames per second on webcam", 30, 30, "0", ">="),
    ]
    for i, (goal, target, result, fmt, op) in enumerate(goals, 5):
        put(sm, i, 1, goal)
        put(sm, i, 2, target, fmt, INPUT)
        put(sm, i, 3, result, fmt, INPUT if isinstance(result, int) else BODY)
        put(sm, i, 4, f'=IF(C{i}{op}B{i},"PASS","FAIL")')
        for c in range(1, 5):
            sm.cell(i, c).border = THIN
    sm["C8"].comment = Comment("Reported by the user from the app's FPS counter (camera capped at 30).", "Helio")
    put(sm, 10, 1, "Other key numbers", font=Font(name=FONT, bold=True))
    others = [
        ("MRL ROC-AUC", "='MRL Test'!B15", "0.000"),
        ("Webcam ROC-AUC", "=Webcam!B15", "0.000"),
        ("Best validation accuracy", f"=Training!C{s}", "0.0%"),
        ("Webcam: closed eyes caught", "=Webcam!B11", "0.0%"),
        ("MRL: open eyes misread as closed", "='MRL Test'!B13", "0.0%"),
        ("Training time (min)", f"=Training!C{s + 2}", "0.0"),
    ]
    for i, (label, f, fmt) in enumerate(others, 11):
        put(sm, i, 1, label)
        put(sm, i, 3, f, fmt)
    put(sm, 18, 1, "Blue = measured or target values; black = formulas pulling from the other tabs.", font=NOTE)
    widths(sm, {"A": 40, "B": 12, "C": 12, "D": 10})

    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = None

    a.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
