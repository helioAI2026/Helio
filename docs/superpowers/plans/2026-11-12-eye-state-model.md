# Eye-State Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a tiny open/closed eye classifier on the MRL Eye Dataset, export it to ONNX, and run it in `edge/` (YuNet face/eye localization → eye crops → classifier → PERCLOS score) next to the existing MediaPipe pipeline on the Mac webcam.

**Architecture:** Two uv projects. `edge/training/` (PyTorch, offline only) owns dataset adapters, subject-split, training, evaluation and ONNX export. `edge/` runtime gains `src/eye_state/` (YuNet wrapper, crop geometry, ONNX classifier, pipeline) using only `opencv-python` + `onnxruntime`; `SleepinessDetector` becomes time-based and signal-agnostic (`"ear"` or `"p_closed"`); `config.DETECTOR` picks `mediapipe | eyestate | compare`.

**Tech Stack:** Python 3.14 (runtime) / 3.12 (training), uv, PyTorch (`mps`), onnx, onnxruntime, OpenCV `FaceDetectorYN`, pytest.

**Spec:** `edge/docs/superpowers/specs/2026-11-12-eye-state-model-design.md`

## Global Constraints

- Inference never imports torch or mediapipe in `eyestate` mode (mediapipe imported lazily only for `mediapipe`/`compare`).
- `eye_state.onnx` < 1 MB, **opset 13**, standard ops only, input grayscale `float32 (N,1,64,64)` in `[0,1]`, output `(N,1)` logit of "closed".
- Train/val/test split **by MRL subject ID**; no subject in more than one split.
- Test-split accuracy ≥ 95%; live webcam eval ≥ 90%; `eyestate` mode ≥ 30 FPS at 640×480 on M4 CPU.
- `p_closed > 0.5` ⇒ closed for PERCLOS; eye term = `max(0, (p_closed - 0.5) / 0.5)`.
- Weights/thresholds in `config.py` unchanged; PERCLOS window = 30 s measured by timestamps.
- MediaPipe mode behaviour unchanged. In `compare`, MediaPipe drives alarm/log/overlay; eyestate is display-only.
- `edge/model/` is gitignored: model files are produced by scripts, never committed.
- UI text stays pt-BR (HUD labels, messages).

## Review Focus

1. **No face / face partly out of frame** (eye crop extends beyond the image) → pipeline returns `None` for that eye, never crashes; both invalid ⇒ `face_detected=False`. Test in Task 5.
2. **Very small face** (inter-eye distance < 12 px) → treated as no valid eyes rather than classifying a few-pixel crop. Test in Task 5.
3. **Frame-rate drops** (compare mode, slow machine) → PERCLOS uses the last 30 s of timestamps, not 900 frames. Test in Task 1.
4. **Model files missing** (`eye_state.onnx` or YuNet not downloaded) → clear pt-BR error naming the script to run, not an onnxruntime stack trace. Test in Task 6.
5. **Running without AWS credentials** → `HELIO_OFFLINE=1` skips MQTT so local testing works. Test in Task 7.

---

## File Structure

```
edge/
  pyproject.toml                       modify: + onnxruntime, [dependency-groups] dev = pytest
  config.py                            modify: DETECTOR, EYE_STATE_MODEL, YUNET_MODEL, P_CLOSED_THRESHOLD
  src/sleepiness_detector.py           modify: time-based, signal-agnostic
  src/alert.py, src/event_logger.py    modify: use result['signal'] / result['eye']
  src/app.py                           modify: mode selection, offline switch, fixed imports
  src/eye_state/__init__.py            create: exports EyeStatePipeline
  src/eye_state/geometry.py            create: eye crop boxes from landmarks
  src/eye_state/yunet.py               create: FaceDetectorYN wrapper
  src/eye_state/classifier.py          create: onnxruntime classifier
  src/eye_state/pipeline.py            create: frame → EyeStateResult
  scripts/download_models.py           create: fetch YuNet into model/
  tests/                               create: test_sleepiness.py, test_geometry.py, test_pipeline.py, test_app_offline.py
  training/                            create: separate uv project
    pyproject.toml, .python-version (3.12)
    eyestate/datasets.py               adapters (MRL, LiveClip) + subject split
    eyestate/model.py                  EyeNet CNN
    eyestate/train.py                  CLI: train → checkpoints/best.pt
    eyestate/export.py                 CLI: .pt → ../model/eye_state.onnx
    eyestate/evaluate.py               CLI: metrics on test split + live clip
    eyestate/download_mrl.py           CLI: fetch + unzip MRL into data/mrl/
    eyestate/record_eval.py            CLI: record labeled webcam clip
    tests/
```

---

### Task 1: Time-based, signal-agnostic SleepinessDetector (+ test setup)

**Files:**
- Modify: `edge/pyproject.toml` (add `onnxruntime>=1.20`; `[dependency-groups] dev = ["pytest>=8"]`; `[tool.pytest.ini_options] pythonpath = ["."]`)
- Modify: `edge/src/sleepiness_detector.py`, `edge/src/alert.py`, `edge/src/event_logger.py`, `edge/config.py`
- Test: `edge/tests/test_sleepiness.py`

**Interfaces:**
- Produces: `SleepinessDetector(signal: Literal["ear","p_closed"], window_seconds: float = config.PERCLOS_WINDOW_SECONDS)`; `eye_term(v) -> float`; `is_closed(v) -> bool`; `compute(value: float, face_detected: bool, t: float) -> dict` with keys `score, alert, status, perclos, eye, signal`. `config.P_CLOSED_THRESHOLD = 0.5`.

- [ ] **Step 1: Write failing tests** in `tests/test_sleepiness.py`:
  - `test_p_closed_open_eyes_score_zero`: 300 frames `value=0.1` at 30 fps → `score == 0.0`, `status == "NORMAL"`.
  - `test_p_closed_eye_term_mirrors_ear`: `SleepinessDetector("p_closed").eye_term(0.75) == pytest.approx(0.5)`, `eye_term(0.4) == 0.0`, `eye_term(1.0) == 1.0`.
  - `test_sustained_closure_reaches_alerta`: 30 s at 30 fps, eyes closed (`p_closed=0.95`) for the last 12 s → `status == "ALERTA"`.
  - `test_perclos_window_is_time_based`: feed 30 s at 30 fps open, then 30 s at **5 fps** closed → `perclos == pytest.approx(1.0, abs=0.05)` (old frame-count window would give ~0.17).
  - `test_ear_signal_unchanged`: `signal="ear"`, `value=0.30` → closed False, score 0; `value=0.10` repeated → eye term `(0.20-0.10)/0.20`.
  - `test_no_face`: `face_detected=False` → `status == "SEM_FACE"`, `score == 0.0`.
- [ ] **Step 2:** `cd edge && uv run pytest tests/test_sleepiness.py -v` → FAIL (signature mismatch).
- [ ] **Step 3: Implement.** Keep a `deque[(t, closed)]`, drop entries with `t < now - window_seconds`. Smoothing (moving average over `EAR_SMOOTHING_FRAMES`) applies to the raw value for both signals. Mapping:
  Expose `eye_term(v: float) -> float` and `is_closed(v: float) -> bool` as public methods.
  - `ear`: `closed = v < EAR_THRESHOLD`; `eye_term = max(0, (EAR_THRESHOLD - v)/EAR_THRESHOLD)`
  - `p_closed`: `closed = v > P_CLOSED_THRESHOLD`; `eye_term = max(0, (v - 0.5)/0.5)`
  Rest of the score/status math unchanged. Update `Alert.draw` to print `f"{result['signal'].upper()}: {result['eye']:.2f}"` and `EventLogger.log_alert` to write `signal` and `eye` instead of `ear`.
- [ ] **Step 4:** run tests → PASS.
- [ ] **Step 5:** Commit `feat: torna o SleepinessDetector baseado em tempo e independente do sinal`.

---

### Task 2: Training project, MRL adapter and subject split

**Files:**
- Create: `edge/training/pyproject.toml` (deps: `torch>=2.5`, `torchvision`, `onnx`, `onnxruntime`, `opencv-python`, `numpy`, `scikit-learn`; dev: `pytest`), `.python-version` = `3.12`, `.gitignore` (`data/`, `checkpoints/`, `.venv/`)
- Create: `edge/training/eyestate/datasets.py`, `edge/training/eyestate/download_mrl.py`
- Test: `edge/training/tests/test_datasets.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Sample(path: Path, label: int, subject: str)` — label 1 = closed, 0 = open.
  - `class DatasetAdapter(Protocol): def samples(self) -> list[Sample]`
  - `MRLAdapter(root: Path)` — parses `sXXXX_IIIII_G_GL_E_R_L_S.png` recursively; eye-state field `E`: `0` closed, `1` open ⇒ label `1 - E`; subject = `sXXXX`.
  - `subject_split(samples, val_frac=0.15, test_frac=0.15, seed=0) -> tuple[list[Sample], list[Sample], list[Sample]]` — splits **subjects**, deterministic.
  - `EyeDataset(samples, train: bool)` (torch Dataset) → `(tensor (1,64,64) float32 in [0,1], label float32)`; grayscale load, resize 64×64; train-time aug: brightness/contrast ±30%, Gaussian blur p=0.3, rotation ±10°, scale 0.9–1.1, hflip p=0.5.
  - `download_mrl.py`: downloads `http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip` to `data/`, unzips to `data/mrl/`; on HTTP failure prints the URL and asks the user to download manually.

- [ ] **Step 1: Failing tests** (build a fake MRL tree in `tmp_path` with tiny PNGs for subjects `s0001..s0010`):
  - `test_mrl_parses_label_and_subject`: `s0003_00012_0_1_0_0_1_01.png` → `label == 1`, `subject == "s0003"`; `..._1_...` in E position → `label == 0`.
  - `test_mrl_ignores_non_png`.
  - `test_subject_split_no_leakage`: subject sets of the three splits are pairwise disjoint and cover all subjects.
  - `test_subject_split_deterministic`: same seed → identical splits.
  - `test_eye_dataset_shape`: item tensor shape `(1, 64, 64)`, dtype float32, `0 <= x <= 1`.
- [ ] **Step 2:** `cd edge/training && uv run pytest -v` → FAIL.
- [ ] **Step 3:** Implement `datasets.py` and `download_mrl.py` as above.
- [ ] **Step 4:** tests → PASS.
- [ ] **Step 5:** Commit `feat: adiciona projeto de treino com adaptador MRL e divisão por sujeito`.

---

### Task 3: EyeNet model and training loop

**Files:**
- Create: `edge/training/eyestate/model.py`, `edge/training/eyestate/train.py`
- Test: `edge/training/tests/test_model_train.py`

**Interfaces:**
- Consumes: `MRLAdapter`, `subject_split`, `EyeDataset` (Task 2).
- Produces:
  - `EyeNet()` — `nn.Module`, input `(N,1,64,64)`, output `(N,1)` logits. 4 blocks of conv3×3-BN-ReLU-maxpool (16, 32, 64, 96 channels) → global avg pool → dropout 0.2 → Linear(96,1). Must stay < 250k params (< 1 MB fp32).
  - `train(data_root: Path, out_dir: Path, epochs: int = 15, batch_size: int = 256, lr: float = 1e-3, device: str | None = None, adapter: DatasetAdapter | None = None) -> Path` — AdamW + cosine LR, `BCEWithLogitsLoss` with `pos_weight` from class balance, picks `mps` → `cuda` → `cpu`, saves best-val-accuracy checkpoint to `out_dir/best.pt` (state_dict + `{"split_seed", "test_subjects"}`), returns its path. CLI: `uv run python -m eyestate.train --data data/mrl --out checkpoints`.

- [ ] **Step 1: Failing tests:**
  - `test_eyenet_output_shape_and_size`: `EyeNet()(torch.zeros(2,1,64,64)).shape == (2,1)`; `sum(p.numel()) < 250_000`.
  - `test_train_smoke`: fake dataset (Task 2 fixture, 10 subjects, ~200 imgs, closed = dark images, open = bright) with `epochs=2, device="cpu"` → returns existing `best.pt`; loaded meta has `test_subjects` non-empty.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** Commit `feat: adiciona EyeNet e loop de treino`.

---

### Task 4: ONNX export, evaluation, and the real training run

**Files:**
- Create: `edge/training/eyestate/export.py`, `edge/training/eyestate/evaluate.py`
- Modify: `edge/training/eyestate/datasets.py` (add `LiveClipAdapter`)
- Test: `edge/training/tests/test_export_eval.py`

**Interfaces:**
- Consumes: `EyeNet`, `train` checkpoint format, adapters.
- Produces:
  - `export(checkpoint: Path, out_path: Path) -> Path` — `torch.onnx.export(opset_version=13, input_names=["input"], output_names=["logit"], dynamic_axes={"input": {0: "n"}, "logit": {0: "n"}})`, then `onnx.checker.check_model`. CLI default out: `../model/eye_state.onnx`.
  - `evaluate(onnx_path: Path, samples: list[Sample]) -> dict` with `accuracy, roc_auc, confusion (2×2 list), n`; CLI prints MRL test-split metrics (subjects from checkpoint meta) and, if `--live data/live` exists, live metrics.
  - `LiveClipAdapter(root: Path)` — label from parent folder name, subject `"live"`.

- [ ] **Step 1: Failing tests:**
  - `test_export_roundtrip`: export a randomly initialised `EyeNet` checkpoint → onnxruntime output matches torch on a random batch of 4 (`atol=1e-4`); `onnx.load(path).opset_import[0].version == 13`; file size < 1_000_000 bytes.
  - `test_evaluate_metrics_keys`: on fake samples → keys present, `0 <= accuracy <= 1`, confusion sums to `n`.
  - `test_live_clip_adapter`: `live/closed/00001.png` → `label == 1`.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** run → PASS.
- [ ] **Step 5:** Commit `feat: adiciona exportação ONNX e avaliação`.
- [ ] **Step 6: Real run (verification):**
  ```bash
  cd edge/training
  uv run python -m eyestate.download_mrl
  uv run python -m eyestate.train --data data/mrl --out checkpoints
  uv run python -m eyestate.export --checkpoint checkpoints/best.pt
  uv run python -m eyestate.evaluate --onnx ../model/eye_state.onnx --checkpoint checkpoints/best.pt
  ```
  Expected: test accuracy ≥ 0.95, `ls -l ../model/eye_state.onnx` < 1 MB. If < 0.95, raise epochs / tweak aug before moving on; report numbers either way.

---

### Task 5: YuNet wrapper and eye-crop geometry

**Files:**
- Create: `edge/src/eye_state/__init__.py`, `geometry.py`, `yunet.py`, `edge/scripts/download_models.py`
- Modify: `edge/config.py` (`YUNET_MODEL = "model/face_detection_yunet_2023mar.onnx"`, `EYE_STATE_MODEL = "model/eye_state.onnx"`, `EYE_CROP_SCALE = 0.6`, `MIN_EYE_DISTANCE_PX = 12`)
- Test: `edge/tests/test_geometry.py`

**Interfaces:**
- Produces:
  - `@dataclass Face(box: np.ndarray, right_eye: np.ndarray, left_eye: np.ndarray, score: float)` (points in pixel `(x,y)`; YuNet row = `x,y,w,h, re_x,re_y, le_x,le_y, nose, rmouth, lmouth, score`).
  - `class YuNet(model_path: str, score_threshold: float = 0.7)`; `detect(bgr: np.ndarray) -> Face | None` — highest-score face; calls `setInputSize` when frame size changes.
  - `eye_boxes(face: Face, frame_shape: tuple[int,int], scale: float = EYE_CROP_SCALE, min_dist: float = MIN_EYE_DISTANCE_PX) -> tuple[Box | None, Box | None]` where `Box = tuple[int,int,int,int]` `(x0,y0,x1,y1)`: square of side `scale * inter_eye_distance` centred on each eye; `None` for a box that leaves the frame; both `None` when distance < `min_dist`.
  - `crop_eye(gray: np.ndarray, box: Box, size: int = 64) -> np.ndarray` float32 `(64,64)` in `[0,1]`.
  - `scripts/download_models.py` fetches `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` into `model/`.

- [ ] **Step 1: Failing tests** (`tests/test_geometry.py`, pure numpy — no model file):
  - `test_boxes_square_centered`: eyes at (100,100),(160,100), frame 480×640 → side 36, centered on each eye.
  - `test_box_out_of_frame_is_none`: right eye at (5,100) → right box `None`, left box valid.
  - `test_tiny_face_both_none`: eyes 8 px apart → `(None, None)`.
  - `test_crop_eye_shape_range`: shape `(64,64)`, float32, within `[0,1]`.
- [ ] **Step 2:** `cd edge && uv run pytest tests/test_geometry.py -v` → FAIL. **Step 3:** implement. **Step 4:** PASS.
- [ ] **Step 5:** `uv run python scripts/download_models.py` → file exists; manual check: `YuNet(...).detect` on a webcam frame returns a `Face`.
- [ ] **Step 6:** Commit `feat: adiciona YuNet e geometria de recorte dos olhos`.

---

### Task 6: ONNX classifier and EyeStatePipeline

**Files:**
- Create: `edge/src/eye_state/classifier.py`, `edge/src/eye_state/pipeline.py`
- Test: `edge/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `YuNet`, `Face`, `eye_boxes`, `crop_eye` (Task 5); ONNX contract from Global Constraints.
- Produces:
  - `class EyeStateClassifier(model_path: str)`; `predict(crops: np.ndarray (N,64,64)) -> np.ndarray (N,)` probabilities (sigmoid of logit). `CPUExecutionProvider`.
  - `@dataclass EyeStateResult(face: Face | None, boxes: tuple[Box|None, Box|None], p_left: float | None, p_right: float | None, p_closed: float | None)`; `face_detected` property = `p_closed is not None`.
  - `class EyeStatePipeline(yunet_path: str, classifier_path: str)`; `process(rgb_float_frame: np.ndarray) -> EyeStateResult` (accepts the `Camera.read()` format: RGB float32 `[0,1]`). Both eyes batched in one `predict` call; `p_closed` = mean of valid eyes.
  - `ModelMissingError(Exception)` raised at construction when a file is absent, message: `f"Modelo não encontrado: {path}. Rode {hint}"` with hint `scripts/download_models.py` for YuNet and `training: python -m eyestate.export` for the classifier.

- [ ] **Step 1: Failing tests** — fixture writes a tiny ONNX in `tmp_path` with `onnx.helper` (input `(N,1,64,64)` → `ReduceMean` → output `(N,1)`, opset 13; add `onnx` to edge dev group) so tests don't need the trained model; fake `YuNet` via monkeypatch returning a fixed `Face`:
  - `test_classifier_predict_shape`: 2 crops → shape `(2,)`, values in `(0,1)`.
  - `test_pipeline_mean_of_both_eyes` and `test_pipeline_one_eye_valid_uses_it`.
  - `test_pipeline_no_face`: fake YuNet returns `None` → `p_closed is None`, `face_detected is False`.
  - `test_missing_model_message`: nonexistent path → `ModelMissingError` with `"download_models.py"` in message.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: adiciona classificador ONNX e pipeline de estado dos olhos`.

---

### Task 7: App integration — modes, compare HUD, offline switch

**Files:**
- Modify: `edge/config.py` (`DETECTOR = "eyestate"`), `edge/src/app.py`, `edge/src/video.py` (draw eye boxes), `edge/src/alert.py` (compare panel)
- Modify: imports in `src/app.py`, `src/helpers/aws.py` to `src.helpers…` / `src.models…` (currently `python -c "import src.app"` fails with `No module named 'helpers'`)
- Test: `edge/tests/test_app_offline.py`

**Interfaces:**
- Consumes: `SleepinessDetector` (Task 1), `EyeStatePipeline` (Task 6), existing `FaceDetector`.
- Produces: `App(detector: str = config.DETECTOR)`; `HELIO_OFFLINE=1` env var skips MQTT construction and the startup test event (prints `"Modo offline: MQTT desativado"`).

Behaviour:
- `mediapipe`: exactly today's flow, with `SleepinessDetector("ear")` and `t = time.monotonic()`.
- `eyestate`: `EyeStatePipeline` → `SleepinessDetector("p_closed")` drives alert/log/overlay; `Display.render` draws the two eye boxes (green/red by drowsy) instead of landmarks; mediapipe is **not imported**.
- `compare`: both run on the same frame; MediaPipe result drives alert/log/overlay; HUD shows a second line `"MODELO: <status> <score%> P_FECHADO: <p>"` from the eyestate detector; both overlays drawn. HUD shows measured FPS in all modes.

- [ ] **Step 1: Failing tests:**
  - `test_offline_skips_mqtt`: monkeypatch `HELIO_OFFLINE=1`, patch `Camera`/`Display`/`EyeStatePipeline` with fakes, patch `src.helpers.aws.MQTT` to raise if constructed → `App("eyestate")` constructs fine.
  - `test_eyestate_mode_does_not_import_mediapipe`: after constructing `App("eyestate")` (with fakes), `"mediapipe" not in sys.modules` (run in subprocess for a clean module table).
  - `test_unknown_detector_raises`: `App("foo")` → `ValueError`.
- [ ] **Step 2:** FAIL. **Step 3:** implement (lazy `from src.face_detector import FaceDetector` inside the mediapipe/compare branch). **Step 4:** PASS, plus full `uv run pytest` green.
- [ ] **Step 5: Manual verification** (`HELIO_OFFLINE=1 uv run main.py` with each `DETECTOR`): eyestate FPS ≥ 30 on HUD; closing eyes flips `P_FECHADO` > 0.5 within a few frames; keeping eyes mostly closed ~10 s reaches AVISO/ALERTA; mediapipe mode looks as before; compare shows both.
- [ ] **Step 6:** Commit `feat: integra modelo de estado dos olhos ao app com modos mediapipe/eyestate/compare`.

---

### Task 8: Live webcam evaluation

**Files:**
- Create: `edge/training/eyestate/record_eval.py`
- Test: `edge/training/tests/test_record_eval.py`

**Interfaces:**
- Consumes: `YuNet`, `eye_boxes`, `crop_eye` from `edge/src/eye_state` (Task 5; imported by inserting `edge/` on `sys.path`), `evaluate`/`LiveClipAdapter` (Task 4).
- Produces: `save_crop(crop: np.ndarray, out: Path, label: str) -> Path` writing `out/{open,closed}/NNNNN.png` (next free index). CLI: webcam loop that saves both eye crops each frame while `o` (open) or `c` (closed) is held; `q` quits; shows counts on screen.

- [ ] **Step 0:** Test `test_save_crop_numbering`: two saves to `closed` → `00000.png`, `00001.png`, readable by `LiveClipAdapter` with `label == 1`. FAIL → implement → PASS → commit `feat: adiciona gravação de clipe de avaliação ao vivo`.
- [ ] **Step 1:** `cd edge/training && uv run python -m eyestate.record_eval --out data/live` — user records ~1 min (half open, half closed; with and without glasses if available).
- [ ] **Step 2:** `uv run python -m eyestate.evaluate --onnx ../model/eye_state.onnx --checkpoint checkpoints/best.pt --live data/live` → expected live accuracy ≥ 0.90.
- [ ] **Step 3 (only if < 0.90):** add `CEWAdapter` (Closed Eyes in the Wild, eye-patch release; folder name → label, subject = file stem prefix) to `datasets.py` with a parse test, train on MRL+CEW, pick the threshold maximizing live accuracy and put it in `config.P_CLOSED_THRESHOLD`; re-run Steps 2 and Task 7 Step 5.
- [ ] **Step 4:** Add a short "Modelo de estado dos olhos" section to a new `edge/README.md` (pt-BR): download models, train, export, evaluate, run modes, `HELIO_OFFLINE`. Commit `docs: documenta o treino e o uso do modelo de estado dos olhos`.
