# Helio — Driver Drowsiness Detection

Real-time drowsiness detection system for drivers using webcam and facial landmark analysis.

## Overview

Helio monitors the driver's face through a webcam and continuously computes a drowsiness score based on two complementary metrics:

- **EAR (Eye Aspect Ratio)** — measures the eye openness ratio frame by frame (fast reaction)
- **PERCLOS (Percentage of Eye Closure)** — measures the proportion of time eyes remained closed over a 30-second sliding window (robust long-term indicator)

The final score combines both: `score = 0.3 × EAR_score + 0.7 × PERCLOS_score`

### Alert States

| Status | Score | Visual |
|--------|-------|--------|
| NORMAL | < 30% | Green HUD |
| AVISO  | 30–60% | Orange HUD |
| ALERTA | ≥ 60% | Red HUD + red screen overlay + audio alarm |

## Requirements

- Python 3.14+
- Webcam
- [`uv`](https://docs.astral.sh/uv/) (recommended package manager)

## Installation

```bash
git clone https://github.com/helioAI2026/Helio.git
cd Helio
uv sync
```

## Usage

```bash
uv run python main.py
```

Press `Q` to quit.

## Configuration

All parameters are in [config.py](config.py). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EAR_THRESHOLD` | 0.20 | EAR value below which eyes are considered closed |
| `PERCLOS_WINDOW_SECONDS` | 30 | Sliding window duration for PERCLOS calculation |
| `PERCLOS_THRESHOLD` | 0.20 | PERCLOS ratio that begins to contribute to the score |
| `EAR_WEIGHT` / `PERCLOS_WEIGHT` | 0.3 / 0.7 | Score weighting between EAR and PERCLOS |
| `CAMERA_ID` | 0 | Camera index (0 = default webcam) |

## Project Structure

```
Helio/
├── main.py              # Entry point
├── config.py            # All tunable parameters
├── model/
│   └── face_landmarker.task  # MediaPipe face landmark model
└── src/
    ├── app.py               # Main loop and orchestration
    ├── face_detector.py     # MediaPipe landmark extraction + EAR calculation
    ├── sleepiness_detector.py  # PERCLOS + score computation
    ├── alert.py             # HUD rendering and overlay
    ├── video.py             # Camera capture and display
    └── event_logger.py      # Session log persistence
```

## How It Works

1. Each frame is passed to `FaceDetector`, which uses MediaPipe's Face Landmarker to extract 478 facial landmarks.
2. Six landmarks per eye are used to compute EAR via the formula: `EAR = (||p2–p6|| + ||p3–p5||) / (2 × ||p1–p4||)`.
3. `SleepinessDetector` maintains a rolling buffer of EAR values over the configured window and derives PERCLOS.
4. The weighted score determines the alert state, which `Alert` renders as an on-screen HUD with a progress bar.
5. Alert events are persisted to a session log file by `EventLogger`.

## Dependencies

| Library | Purpose |
|---------|---------|
| `opencv-python` | Camera capture and frame rendering |
| `mediapipe` | Face landmark detection model |
| `numpy` | Landmark coordinate math |
| `scipy` | Signal processing utilities |
