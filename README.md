# Helio — Edge

Detector de sonolência que roda no dispositivo (hoje no MacBook; depois no Jetson Nano).

## Modos (`DETECTOR` em `config.py`)

| Modo        | O que roda                                                                          |
| ----------- | ----------------------------------------------------------------------------------- |
| `mediapipe` | Pipeline original: MediaPipe Face Landmarker → EAR → PERCLOS                        |
| `eyestate`  | YuNet (rosto + olhos) → classificador próprio aberto/fechado (`p_closed`) → PERCLOS |
| `compare`   | Os dois no mesmo quadro; o MediaPipe controla alarme/log, o modelo aparece no HUD   |

```bash
uv sync
uv run python scripts/download_models.py   # YuNet → model/
HELIO_OFFLINE=1 uv run main.py              # sem AWS/MQTT; "q" sai
uv run pytest
```

`HELIO_OFFLINE=1` pula a conexão MQTT (útil sem credenciais AWS). No macOS, o terminal precisa
de permissão de câmera (Ajustes do Sistema → Privacidade e Segurança → Câmera).

## Modelo de estado dos olhos (`training/`)

Projeto `uv` separado (PyTorch fica fora do runtime). Dataset: **MRL Eye Dataset** (imagens de
olhos em infravermelho, uso para pesquisa), dividido **por pessoa** em treino/validação/teste.

```bash
cd training
uv sync
uv run python -m eyestate.download_mrl     # ou baixe o zip manualmente para data/mrlEyes_2018_01.zip
uv run python -m eyestate.train --data data/mrl --out checkpoints
uv run python -m eyestate.export --checkpoint checkpoints/best.pt   # → ../model/eye_state.onnx
uv run python -m eyestate.evaluate --onnx ../model/eye_state.onnx --checkpoint checkpoints/best.pt
```

Avaliação na sua webcam (só avaliação, não treino):

```bash
uv run python -m eyestate.record_eval --out data/live    # "o" aberto, "c" fechado, espaço pausa, "q" sai
uv run python -m eyestate.evaluate --onnx ../model/eye_state.onnx --checkpoint checkpoints/best.pt --live data/live
```

Metas: ≥ 95% de acurácia nas pessoas de teste do MRL, ≥ 90% no clipe da webcam, modelo < 1 MB,
ONNX opset 13 (convertível para TensorRT no Jetson Nano).

## Status (2026-11-12)

Pipeline construído e testado (`uv run pytest` no edge e em `training/`), mas as metas acima
**ainda não foram verificadas**:

- [ ] treino real no MRL (o download foi bloqueado pela rede; baixe o zip manualmente)
- [ ] ≥ 95% nos sujeitos de teste do MRL
- [ ] ≥ 90% no clipe da webcam (`record_eval`)
- [ ] ≥ 30 FPS na webcam (medido 120 FPS do pipeline sem câmera, com pesos não treinados)

Até existir `model/eye_state.onnx`, o modo `eyestate` encerra com uma mensagem explicando como
gerá-lo; use `DETECTOR = "mediapipe"` enquanto isso.

Design: `docs/superpowers/specs/2026-11-12-eye-state-model-design.md`.
