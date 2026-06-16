import time
from collections import deque
import config


class SleepinessDetector:
    def __init__(self, ear_threshold: float = None, perclos_window: float = None):
        self.ear_threshold = ear_threshold or config.EAR_THRESHOLD
        self.perclos_threshold = config.PERCLOS_THRESHOLD
        self.ear_weight = config.EAR_WEIGHT
        self.perclos_weight = config.PERCLOS_WEIGHT
        self.status_normal_threshold = config.STATUS_NORMAL_THRESHOLD
        self.status_aviso_threshold = config.STATUS_AVISO_THRESHOLD
        self.blink_frames_threshold = config.BLINK_FRAMES_THRESHOLD

        self.perclos_window = perclos_window or config.PERCLOS_WINDOW_SECONDS
        self._perclos_buffer: deque[tuple[float, bool]] = deque()
        self._ear_history: deque[float] = deque(maxlen=config.EAR_SMOOTHING_FRAMES)
        self._blink_streak = 0

    def compute(self, ear: float, face_detected: bool = True) -> dict:
        if not face_detected:
            self._blink_streak = 0
            return {'score': 0.0, 'alert': False, 'ear': 0.0, 'perclos': 0.0, 'status': 'SEM_FACE'}

        now = time.monotonic()

        self._ear_history.append(ear)
        smoothed_ear = sum(self._ear_history) / len(self._ear_history)

        if smoothed_ear < self.ear_threshold:
            self._blink_streak += 1
        else:
            self._blink_streak = 0

        is_closed = smoothed_ear < self.ear_threshold and self._blink_streak > self.blink_frames_threshold
        self._perclos_buffer.append((now, is_closed))

        cutoff = now - self.perclos_window
        while self._perclos_buffer and self._perclos_buffer[0][0] < cutoff:
            self._perclos_buffer.popleft()

        perclos = sum(1 for _, c in self._perclos_buffer if c) / len(self._perclos_buffer) if self._perclos_buffer else 0.0

        ear_score = max(0, (self.ear_threshold - smoothed_ear) / self.ear_threshold)
        perclos_score = max(0, (perclos - self.perclos_threshold) / (0.40 - self.perclos_threshold))

        score = self.ear_weight * ear_score + self.perclos_weight * perclos_score
        score = min(max(0, score), 1.0)

        status = 'NORMAL' if score < self.status_normal_threshold else ('AVISO' if score < self.status_aviso_threshold else 'ALERTA')

        return {
            'score': score,
            'alert': score >= 0.60,
            'ear': smoothed_ear,
            'perclos': perclos,
            'status': status,
        }

    def reset(self):
        self._perclos_buffer.clear()
        self._ear_history.clear()
        self._blink_streak = 0