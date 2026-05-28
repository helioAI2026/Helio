from collections import deque
import config


class SleepinessDetector:
    def __init__(self, ear_threshold: float = None, fps: float = None, perclos_window: float = None):
        self.ear_threshold = ear_threshold or config.EAR_THRESHOLD
        self.perclos_threshold = config.PERCLOS_THRESHOLD
        self.ear_weight = config.EAR_WEIGHT
        self.perclos_weight = config.PERCLOS_WEIGHT
        self.status_normal_threshold = config.STATUS_NORMAL_THRESHOLD
        self.status_aviso_threshold = config.STATUS_AVISO_THRESHOLD
        
        fps = fps or config.CAMERA_FPS
        perclos_window = perclos_window or config.PERCLOS_WINDOW_SECONDS
        self.window_frames = int(fps * perclos_window)
        self._perclos_buffer: deque[bool] = deque(maxlen=self.window_frames)
        self._ear_history: deque[float] = deque(maxlen=config.EAR_SMOOTHING_FRAMES)

    def compute(self, ear: float, face_detected: bool = True) -> dict:
        if not face_detected:
            return {'score': 0.0, 'alert': False, 'ear': 0.0, 'perclos': 0.0, 'status': 'SEM_FACE'}

        # armazena EAR bruto e calcula média móvel
        self._ear_history.append(ear)
        smoothed_ear = sum(self._ear_history) / len(self._ear_history)
        
        # registra se olhos estão fechados (EAR < threshold) e calcula PERCLOS (% fechamento em 30s)
        self._perclos_buffer.append(smoothed_ear < self.ear_threshold)
        perclos = sum(self._perclos_buffer) / len(self._perclos_buffer) if self._perclos_buffer else 0.0
        
        # normaliza EAR: quanto menor o valormaior o score (0.0 aberto e 1.0 completamente fechado)
        ear_score = max(0, (self.ear_threshold - smoothed_ear) / self.ear_threshold)
        # normaliza PERCLOS: quanto maior o %, maior o score
        perclos_score = max(0, (perclos - self.perclos_threshold) / (0.40 - self.perclos_threshold))
        
        # Combina EAR (40%) + PERCLOS (60%) para score final (0.0-1.0)
        score = self.ear_weight * ear_score + self.perclos_weight * perclos_score
        score = min(max(0, score), 1.0)
        
        # Classifica status: NORMAL (<30%), AVISO (30-60%), ALERTA (>=60%)
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
        self._face_lost_frames = 0
