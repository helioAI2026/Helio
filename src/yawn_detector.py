from collections import deque
from scipy.spatial import distance


class YawnDetector:

    LEFT_CORNER  = 61
    RIGHT_CORNER = 291
    UPPER_LEFT   = 39
    UPPER_CENTER = 13
    UPPER_RIGHT  = 269
    LOWER_LEFT   = 375
    LOWER_CENTER = 14
    LOWER_RIGHT  = 405

    # Valores padrão
    DEFAULT_MAR_THRESHOLD  = 0.55   # MAR acima disso = boca muito aberta
    DEFAULT_CONSEC_FRAMES  = 45     # ~1.5 s a 30 fps para confirmar bocejo
    DEFAULT_COOLDOWN       = 90     # ~3 s entre alertas consecutivos
    DEFAULT_HISTORY        = 300    # últimos 10 s para frequência de bocejos

    def __init__(
        self,
        mar_threshold: float = DEFAULT_MAR_THRESHOLD,
        consec_frames: int   = DEFAULT_CONSEC_FRAMES,
        cooldown:      int   = DEFAULT_COOLDOWN,
        history:       int   = DEFAULT_HISTORY,
    ):
        self.mar_threshold = mar_threshold
        self.consec_frames = consec_frames
        self.cooldown      = cooldown

        self._counter  = 0          # frames consecutivos com MAR alto
        self._since    = cooldown   # frames desde o último alerta
        self._yawning  = False      # estava bocejando no frame anterior
        self._yawn_log: deque[bool] = deque(maxlen=history)  # True = frame de bocejo

    
    def extract(self, landmarks) -> list:
        idx = [
            self.LEFT_CORNER,
            self.RIGHT_CORNER,
            self.UPPER_LEFT,
            self.UPPER_CENTER,
            self.UPPER_RIGHT,
            self.LOWER_LEFT,
            self.LOWER_CENTER,
            self.LOWER_RIGHT,
        ]
        return [landmarks[i] for i in idx]

    
    def _mar(self, mouth: list) -> float:
        left_corner, right_corner = mouth[0], mouth[1]
        upper_left, upper_center, upper_right = mouth[2], mouth[3], mouth[4]
        lower_left, lower_center, lower_right = mouth[5], mouth[6], mouth[7]

        A = distance.euclidean(upper_left,   lower_left)    
        B = distance.euclidean(upper_right,  lower_right)   
        D = distance.euclidean(upper_center, lower_center)  
        C = distance.euclidean(left_corner,  right_corner)  

        if C == 0:
            return 0.0

        return (A + B + D) / (3.0 * C)

    
    def update(self, mouth: list | None) -> tuple[float, bool, bool, int]:
        
        if mouth is None:
            self._counter = 0
            self._yawning = False
            self._yawn_log.append(False)
            self._since += 1
            return 0.0, False, False, self._yawn_count()

        mar = self._mar(mouth)

        # conta frames consecutivos com boca muito aberta
        if mar >= self.mar_threshold:
            self._counter += 1
        else:
            self._counter = 0

        self._since += 1

        yawning = self._counter >= self.consec_frames

        # detecta início de um novo bocejo (transição False → True)
        new_yawn = yawning and not self._yawning
        self._yawning = yawning

        # registra no histórico de frequência
        self._yawn_log.append(yawning)

        # alerta respeita cooldown e só dispara no início do bocejo
        yawn_alert = new_yawn and self._since >= self.cooldown
        if yawn_alert:
            self._since = 0

        return mar, yawning, yawn_alert, self._yawn_count()

    
    def _yawn_count(self) -> int:
        
        count = 0
        prev = False
        for v in self._yawn_log:
            if v and not prev:
                count += 1
            prev = v
        return count

    
    def reset(self) -> None:
        self._counter = 0
        self._since   = self.cooldown
        self._yawning = False
        self._yawn_log.clear()