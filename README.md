# Helio — Detecção de Sonolência ao Volante

Sistema de detecção de sonolência em tempo real para motoristas, utilizando webcam e análise de landmarks faciais.

## Visão Geral

O Helio monitora o rosto do motorista pela webcam e calcula continuamente um score de sonolência baseado em duas métricas complementares:

- **EAR (Eye Aspect Ratio)** — mede a proporção de abertura dos olhos quadro a quadro (reação rápida)
- **PERCLOS (Percentage of Eye Closure)** — mede o percentual de tempo com os olhos fechados em uma janela deslizante de 30 segundos (indicador robusto de longo prazo)

O score final combina as duas métricas: `score = 0.3 × score_EAR + 0.7 × score_PERCLOS`

### Estados de Alerta

| Status | Score | Visual |
|--------|-------|--------|
| NORMAL | < 30% | HUD verde |
| AVISO  | 30–60% | HUD laranja |
| ALERTA | ≥ 60% | HUD vermelho + overlay de tela vermelha + alarme sonoro |

## Requisitos

- Python 3.9+ (veja restrições por plataforma abaixo)
- Webcam
- [`uv`](https://docs.astral.sh/uv/) (gerenciador de pacotes recomendado, para x86_64)
- `ffmpeg` (para o alerta sonoro)

## Instalação

### x86_64 (Linux/macOS/Windows) — Python 3.11+

```bash
git clone https://github.com/helioAI2026/Helio.git
cd Helio
uv sync
```

### ARM64 — Debian 11 (Bullseye) — Tinker Board 2S / dispositivos com glibc 2.31

O `tflite-runtime` não oferece wheel para Python 3.11+ no Debian 11 (glibc 2.31). Use o script de instalação dedicado, que cria um virtualenv com Python 3.9 e fixa as versões compatíveis:

```bash
git clone https://github.com/helioAI2026/Helio.git
cd Helio
bash install-arm64.sh
```

## Uso

### x86_64

```bash
uv run python main.py
```

### ARM64

```bash
.venv-arm64/bin/python main.py
```

Pressione `Q` para sair.

## Configuração

Todos os parâmetros estão em [config.py](config.py). Principais configurações:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `EAR_THRESHOLD` | 0.20 | Valor de EAR abaixo do qual os olhos são considerados fechados |
| `PERCLOS_WINDOW_SECONDS` | 30 | Duração da janela deslizante para cálculo do PERCLOS |
| `PERCLOS_THRESHOLD` | 0.20 | Proporção de PERCLOS que começa a contribuir para o score |
| `EAR_WEIGHT` / `PERCLOS_WEIGHT` | 0.3 / 0.7 | Pesos do score entre EAR e PERCLOS |
| `CAMERA_ID` | 0 | Índice da câmera (0 = webcam padrão) |

## Estrutura do Projeto

```
Helio/
├── main.py              # Ponto de entrada
├── config.py            # Todos os parâmetros configuráveis
├── model/
│   ├── face_detection.tflite  # Modelo de detecção de face
│   └── face_landmark.tflite   # Modelo de landmarks faciais
└── src/
    ├── app.py               # Loop principal e orquestração
    ├── face_detector.py     # Extração de landmarks via TFLite + cálculo do EAR
    ├── sleepiness_detector.py  # Cálculo do PERCLOS e score de sonolência
    ├── alert.py             # Renderização do HUD e overlay
    ├── video.py             # Captura de câmera e exibição
    └── event_logger.py      # Persistência do log de sessão
```

## Como Funciona

1. Cada frame é passado ao `FaceDetector`, que usa dois modelos TFLite (`face_detection.tflite` + `face_landmark.tflite`) para detectar o rosto e extrair landmarks faciais.
2. Seis landmarks por olho são usados para calcular o EAR pela fórmula: `EAR = (||p2–p6|| + ||p3–p5||) / (2 × ||p1–p4||)`.
3. O `SleepinessDetector` mantém um buffer circular de valores de EAR na janela configurada e deriva o PERCLOS.
4. O score ponderado determina o estado de alerta, que o `Alert` renderiza como HUD com barra de progresso na tela.
5. Eventos de alerta são persistidos em um arquivo de log de sessão pelo `EventLogger`.

## Dependências

| Biblioteca | Finalidade |
|------------|------------|
| `opencv-python` | Captura de câmera e renderização de frames |
| `tflite-runtime` | Inferência dos modelos de detecção facial e landmarks |
| `numpy` | Cálculos com coordenadas dos landmarks |
| `scipy` | Utilitários de processamento de sinais |
| `ffmpeg` (sistema) | Reprodução do alerta sonoro via `ffplay` |
