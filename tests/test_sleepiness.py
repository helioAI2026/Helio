import pytest

import config
from src.sleepiness_detector import SleepinessDetector


def feed(det, value, seconds, fps, t0=0.0):
    r = None
    n = int(seconds * fps)
    for i in range(n):
        r = det.compute(value, True, t0 + i / fps)
    return r, t0 + n / fps


def test_p_closed_open_eyes_score_zero():
    det = SleepinessDetector("p_closed")
    r, _ = feed(det, 0.1, 10, 30)
    assert r["score"] == 0.0
    assert r["status"] == "NORMAL"
    assert r["signal"] == "p_closed"


def test_p_closed_eye_term_mirrors_ear():
    det = SleepinessDetector("p_closed")
    assert det.eye_term(0.75) == pytest.approx(0.5)
    assert det.eye_term(0.4) == 0.0
    assert det.eye_term(1.0) == 1.0


def test_sustained_closure_reaches_alerta():
    det = SleepinessDetector("p_closed")
    _, t = feed(det, 0.05, 18, 30)
    r, _ = feed(det, 0.95, 12, 30, t0=t)
    assert r["status"] == "ALERTA"


def test_perclos_window_is_time_based():
    det = SleepinessDetector("p_closed")
    _, t = feed(det, 0.05, 30, 30)
    r, _ = feed(det, 0.95, 30, 5, t0=t)
    assert r["perclos"] == pytest.approx(1.0, abs=0.1)


def test_ear_signal_unchanged():
    det = SleepinessDetector("ear")
    assert det.is_closed(0.30) is False
    assert det.eye_term(0.30) == 0.0
    assert det.is_closed(0.10) is True
    thr = config.EAR_THRESHOLD
    assert det.eye_term(0.10) == pytest.approx((thr - 0.10) / thr)
    r, _ = feed(det, 0.30, 5, 30)
    assert r["score"] == 0.0 and r["signal"] == "ear"


def test_no_face():
    det = SleepinessDetector("p_closed")
    r = det.compute(0.9, False, 0.0)
    assert r["status"] == "SEM_FACE"
    assert r["score"] == 0.0
