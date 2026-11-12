from collections import deque
from typing import Literal

import config

Signal = Literal["ear", "p_closed"]


class SleepinessDetector:
    """Pontuação de sonolência a partir de um sinal por quadro.

    signal="ear":      valor = Eye Aspect Ratio (baixo = olho fechado)
    signal="p_closed": valor = probabilidade de olho fechado do modelo (alto = fechado)

    O PERCLOS é calculado sobre os últimos `window_seconds` segundos (por tempo,
    não por número de quadros), então continua correto quando o FPS varia.
    """

    def __init__(self, signal: Signal, window_seconds: float | None = None):
        if signal not in ("ear", "p_closed"):
            raise ValueError(f"Sinal desconhecido: {signal}")
        self.signal = signal
        self.window_seconds = window_seconds or config.PERCLOS_WINDOW_SECONDS
        self._closed: deque[tuple[float, bool]] = deque()
        self._history: deque[float] = deque(maxlen=config.EAR_SMOOTHING_FRAMES)

    def is_closed(self, v: float) -> bool:
        if self.signal == "ear":
            return v < config.EAR_THRESHOLD
        return v > config.P_CLOSED_THRESHOLD

    def eye_term(self, v: float) -> float:
        if self.signal == "ear":
            thr = config.EAR_THRESHOLD
            return max(0.0, (thr - v) / thr)
        thr = config.P_CLOSED_THRESHOLD
        return min(1.0, max(0.0, (v - thr) / (1.0 - thr)))

    def compute(self, value: float, face_detected: bool, t: float) -> dict:
        if not face_detected:
            return {'score': 0.0, 'alert': False, 'eye': 0.0, 'perclos': 0.0,
                    'status': 'SEM_FACE', 'signal': self.signal}

        self._history.append(value)
        smoothed = sum(self._history) / len(self._history)

        self._closed.append((t, self.is_closed(smoothed)))
        while self._closed and self._closed[0][0] < t - self.window_seconds:
            self._closed.popleft()
        perclos = sum(c for _, c in self._closed) / len(self._closed)

        perclos_score = max(0.0, (perclos - config.PERCLOS_THRESHOLD) / (0.40 - config.PERCLOS_THRESHOLD))
        score = config.EAR_WEIGHT * self.eye_term(smoothed) + config.PERCLOS_WEIGHT * perclos_score
        score = min(max(0.0, score), 1.0)

        if score < config.STATUS_NORMAL_THRESHOLD:
            status = 'NORMAL'
        elif score < config.STATUS_AVISO_THRESHOLD:
            status = 'AVISO'
        else:
            status = 'ALERTA'

        return {
            'score': score,
            'alert': score >= config.STATUS_AVISO_THRESHOLD,
            'eye': smoothed,
            'perclos': perclos,
            'status': status,
            'signal': self.signal,
        }

    def reset(self):
        self._closed.clear()
        self._history.clear()
