# Camera
CAMERA_ID     = 0
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30

# Janela
WINDOW_TITLE  = "Driver Monitor"
WINDOW_WIDTH  = 720
WINDOW_HEIGHT = 540

# Modelo
FACE_MODEL = "model/face_landmarker.task"

# EAR — sonolência instantânea
EAR_THRESHOLD  = 0.20
CONSEC_FRAMES  = 20
ALERT_COOLDOWN = 30

# PERCLOS — tendência acumulada 
PERCLOS_WINDOW      = 30    # segundos
PERCLOS_ALERT_RATIO = 0.15  # 15%

# Bocejo — MAR
MAR_THRESHOLD     = 0.55   # boca muito aberta
YAWN_CONSEC_FRAMES = 45    # ~1.5 s para confirmar bocejo
YAWN_COOLDOWN      = 90    # ~3 s entre alertas
YAWN_HISTORY       = 300   # últimos 10 s para contagem de frequência