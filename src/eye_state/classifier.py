import numpy as np
import onnxruntime as ort


class EyeStateClassifier:
    """eye_state.onnx: input (N,1,64,64) gray in [0,1] -> logit of 'closed' (N,1)."""

    def __init__(self, model_path: str):
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def predict(self, crops: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(crops, dtype=np.float32)[:, None]
        logit = self._session.run(None, {self._input: x})[0][:, 0]
        return 1.0 / (1.0 + np.exp(-logit))
