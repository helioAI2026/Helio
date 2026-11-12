# Eye-state model for Helio edge — design

Date: 2026-11-12
Status: draft, awaiting approval

## Intent and why

Helio's edge unit will be a **Jetson Nano** in the truck cab. Today `edge/` detects drowsiness
with MediaPipe's Face Landmarker: eye landmarks → EAR (eye aspect ratio) → PERCLOS over 30 s →
0–1 score (`src/sleepiness_detector.py`). MediaPipe has no practical GPU path on the Nano
(JetPack 4.x, old Python/CUDA), and landmark-based EAR is fragile on infrared images, glasses,
and head turns.

We want a small vision pipeline we own, trained on public data, that the user can **test locally on
an M4 MacBook (16 GB) with the built-in webcam now**, and that is built so it can later move to the
Nano (ONNX → TensorRT) without redesign.

## Goals

1. **Training pipeline** that downloads a public dataset, trains a tiny open/closed eye
   classifier, evaluates it, and exports it to ONNX.
2. **Inference pipeline** in `edge/` that uses only ONNX/OpenCV (no MediaPipe):
   face + eye localization → eye crops → eye-state classifier → `p_closed` per frame.
3. **Plug into the existing scoring**: `SleepinessDetector` keeps PERCLOS and the
   NORMAL/AVISO/ALERTA logic, but is fed the classifier's closed probability instead of EAR.
4. **Comparison mode** on the Mac webcam: run the new pipeline and the current MediaPipe/EAR
   pipeline on the same frames and show both results so the user can judge them.

## Non-goals (for now)

- Deploying to the Jetson Nano (TensorRT conversion, JetPack packaging, Python 3.6 compatibility).
- Yawn detection, head pose, phone/distraction detection.
- Dashboard or infra changes (devices reporting model version, etc.) — that is the next project.
- Training on the user's own IR camera footage (none exists yet).
- Greengrass retraining loop.

## Anti-goals (would be a failure even if it "works")

- A model that only scores well because the same people appear in train and test
  (split must be **by subject**).
- Anything that can't export to plain ONNX with standard ops, or that needs a big runtime
  (no PyTorch, no MediaPipe at inference time).
- Breaking the current MediaPipe path — it stays available until the user decides to drop it.

## Constraints

- Dev machine: Apple M4, 16 GB RAM, macOS. Train on PyTorch with the `mps` backend; CPU fallback ok.
- Edge project uses Python 3.14 and `uv` (`edge/pyproject.toml`).
- Model must be tiny: target **< 1 MB** ONNX, input **grayscale** (the truck camera is IR; the Mac
  webcam is RGB and is converted to gray), fixed input size (proposed 64×64 — builder's call).
- ONNX opset **≤ 13** so the file stays convertible with the TensorRT 8.0 that ships with JetPack 4.6.
- Real-time on the Mac: full pipeline ≥ 30 FPS at 640×480 on CPU via `onnxruntime`.

## Decisions

Decided with the user:
- Vision model for Helio (not a chat LLM). Start from **public datasets only**.
- Test locally on the MacBook first; Jetson Nano is the eventual target.

Claude's calls (user can override):
- **Dataset: MRL Eye Dataset** (~85k infrared eye crops, 37 subjects, labeled open/closed, with
  glasses/reflection/lighting metadata). Closest public match to an IR cab camera.
  Split train/val/test **by subject ID** (parsed from file names).
  *Licensing:* MRL is distributed for research use. The user confirmed this project is
  research/educational only, so MRL is fine.
- **Face + eye localization: OpenCV YuNet** (`face_detection_yunet_2023mar.onnx`, ~230 KB, via
  `cv2.FaceDetectorYN`). It returns a face box **plus 5 landmarks including both eye centers**,
  which the 1 MB Ultra-Light detector in `onboard/` does not. Eye crops are squares centered on
  each eye, sized relative to inter-eye distance.
- **Classifier:** small CNN (a few conv blocks, or MobileNetV3-small-scale), binary output,
  trained with augmentation for brightness/contrast, blur, small rotations/scale, and horizontal
  flip (so one model serves both eyes).
- **Per-frame signal:** `p_closed = mean(left, right)`; if one eye crop is invalid, use the other.
  `SleepinessDetector` treats `p_closed > 0.5` as "closed" for PERCLOS. The eye term in the score
  is `max(0, (p_closed - 0.5) / 0.5)`, mirroring today's EAR term (0 while eyes are open, so
  open eyes add no noise). Weights and thresholds stay in `config.py` and are unchanged: as today,
  a single blink or 1 s closure only nudges the score; AVISO/ALERTA come from sustained closure
  via PERCLOS. Tuning that behaviour is out of scope.
- **PERCLOS window becomes time-based** (timestamps, not a frame count), so it stays correct when
  FPS varies (compare mode, slower hardware). The default stays 30 s.
- **Dataset-agnostic training code:** the loader takes a dataset adapter (file list, label,
  subject ID). MRL is the first adapter; adding CEW or the user's own footage later is a new
  adapter, not a rewrite.
- **Live sanity check:** a small `record_eval` script records ~1 min of the user's own webcam
  with keyboard-labeled open/closed segments, used **only for evaluation** (not training). The
  evaluation script reports accuracy on it next to the MRL test numbers. If live accuracy is
  clearly worse (< 90%), the fallback is to add CEW (RGB, in-the-wild eyes) as a second
  training adapter and tune the 0.5 threshold on the live set.
- **Compare mode:** MediaPipe drives alarm sound, alert log, overlay and cooldown (today's
  behaviour); the new pipeline is display-only. In `eyestate` mode the new pipeline drives them.
  The ≥ 30 FPS target applies to `eyestate` mode only.
- **Code location:**
  - `edge/training/` — its own `uv` project (torch, torchvision, etc. stay out of the runtime deps):
    download script, dataset/split, train, evaluate, export to ONNX.
  - `edge/src/eye_state/` — runtime: YuNet wrapper, eye cropping, ONNX classifier.
  - `edge/model/` — `eye_state.onnx` and the YuNet `.onnx`.
- **Pipeline selection:** `config.py` gets `DETECTOR = "mediapipe" | "eyestate" | "compare"`.
  `compare` shows both results in the HUD (EAR/PERCLOS/score from each) on the same frames.

## Success criteria

- Test-split (held-out subjects) accuracy ≥ 95% and a reported confusion matrix / ROC-AUC.
- `eye_state.onnx` < 1 MB, opset ≤ 13, loads in `onnxruntime` with no custom ops.
- `uv run main.py` with `DETECTOR="eyestate"` runs on the Mac webcam ≥ 30 FPS; the `p_closed`
  readout tracks eyes opening/closing within a few frames, and keeping eyes mostly closed reaches
  AVISO/ALERTA just as the MediaPipe mode does today.
- Accuracy on the user's live webcam eval clip ≥ 90% (else apply the CEW fallback above).
- Existing MediaPipe mode still works unchanged.
- Unit tests for: subject-based split (no leakage), eye-crop geometry, `SleepinessDetector` with
  `p_closed` input, ONNX export round-trip (PyTorch vs onnxruntime outputs match).

## Left to the builder

- Exact CNN architecture, input size, epochs, optimizer, augmentation strengths.
- Crop size multiplier, handling of tiny/partially out-of-frame faces.
- How the training script fetches MRL (official URL; fail with a clear message if unavailable).
- HUD layout for compare mode.

## Known side issue (not in scope)

`App.__init__` sends a test MQTT event on startup and needs AWS credentials; local testing may need
a way to skip it (e.g. an env flag). Builder may add a minimal offline switch if it blocks running.
