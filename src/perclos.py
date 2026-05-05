from collections import deque


class PERCLOS:

    DEFAULT_WINDOW_SECONDS = 30
    DEFAULT_FPS            = 30
    DEFAULT_ALERT_RATIO    = 0.15  
    DEFAULT_MIN_FRAMES     = 30    

    def __init__(
        self,
        ear_threshold:  float,
        fps:            float = DEFAULT_FPS,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        alert_ratio:    float = DEFAULT_ALERT_RATIO,
        min_frames:     int   = DEFAULT_MIN_FRAMES,
    ):
        self.ear_threshold = ear_threshold
        self.alert_ratio   = alert_ratio
        self.min_frames    = min_frames

        window_frames = int(fps * window_seconds)
        self._buffer: deque[bool] = deque(maxlen=window_frames)

    def update(self, ear: float, face_detected: bool) -> tuple[float, bool]:
      
        if not face_detected:
            
            return self._compute(), False

        eye_closed = ear < self.ear_threshold
        self._buffer.append(eye_closed)

        ratio    = self._compute()
        is_alert = ratio >= self.alert_ratio and len(self._buffer) >= self.min_frames

        return ratio, is_alert

    def _compute(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def window_size(self) -> int:
        return self._buffer.maxlen  

    @property
    def frames_collected(self) -> int:
        return len(self._buffer)