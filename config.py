# ==================== CAMERA SETTINGS ====================
CAMERA_ID = 0  # ID da câmera (0 = câmera padrão)
CAMERA_WIDTH = 640  # Largura de captura (pixels)
CAMERA_HEIGHT = 480  # Altura de captura (pixels)
CAMERA_FPS = 30  # Taxa de quadros (frames por segundo)

# ==================== WINDOW DISPLAY ====================
WINDOW_TITLE = "Helio - Detecção de Sonolência"  
WINDOW_WIDTH = 720  # Largura da janela de exibição (pixels)
WINDOW_HEIGHT = 540  # Altura da janela de exibição (pixels)

# ==================== MODEL PATH ====================
FACE_MODEL = "model/face_landmarker.task"  # Caminho do modelo MediaPipe

# ==================== EAR (Eye Aspect Ratio) ====================
EAR_THRESHOLD = 0.20  # Limite para considerar olhos fechados (0.0 fechado → 0.4+ aberto)
EAR_SMOOTHING_FRAMES = 5  # Janela de suavização EAR (média móvel em N frames)

# ==================== PERCLOS (Percentage of Eye Closure) ====================
PERCLOS_THRESHOLD = 0.20  # Limite de PERCLOS (20% = início de alerta)
PERCLOS_WINDOW_SECONDS = 30  # Janela de acumulação (segundos) - quanto mais tempo, mais robusto

# ==================== SCORE CALCULATION ====================
EAR_WEIGHT = 0.3  # Peso do EAR no score final (30%)
PERCLOS_WEIGHT = 0.7  # Peso do PERCLOS no score final (70%)

# ==================== ALERT THRESHOLDS ====================
STATUS_NORMAL_THRESHOLD = 0.3  # Score < 0.3 = NORMAL (verde)
STATUS_AVISO_THRESHOLD = 0.60  # 0.3 <= Score < 0.60 = AVISO (laranja), Score >= 0.60 = ALERTA (vermelho)

# ==================== ALERT BEHAVIOR ====================
ALERT_OVERLAY_DURATION_SECONDS = 1  # Tempo para tela ficar vermelha após entrar em alerta (segundos)
ALERT_COOLDOWN_SECONDS = 5  # Intervalo mínimo entre logs de alerta consecutivos (segundos)

# ==================== SOUND ALERT ====================
SOUND_ALERT_PATH = "assets/faaah.mp3"  # Caminho do arquivo de som
SOUND_ALERT_DELAY_SECONDS = 3    # Tempo em alerta antes de tocar o som (segundos)
SOUND_ALERT_VOLUME = 0.3         # Volume do alarme (0.0 a 1.0)