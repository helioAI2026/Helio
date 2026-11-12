import os
import time

import config
from src.alert import Alert
from src.event_logger import EventLogger
from src.eye_state import EyeStatePipeline
from src.helpers.consts import DEVICE_ID, DEVICE_SECRET
from src.helpers.ride import Ride
from src.sleepiness_detector import SleepinessDetector
from src.video import Camera, Display

MODES = ("mediapipe", "eyestate", "compare")


class App:
    """mediapipe: pipeline original (EAR).
    eyestate:  YuNet + classificador de olhos (p_closed) — sem MediaPipe.
    compare:   os dois no mesmo quadro; o MediaPipe controla alarme, log e overlay.
    """

    def __init__(self, detector: str = config.DETECTOR):
        if detector not in MODES:
            raise ValueError(f"DETECTOR inválido: {detector!r} (use {', '.join(MODES)})")
        self.mode = detector
        self.ride = Ride(device_id=DEVICE_ID, device_secret=DEVICE_SECRET.encode('utf-8'))

        self.mqtt = None
        if os.environ.get("HELIO_OFFLINE") == "1":
            print("Modo offline: MQTT desativado")
        else:
            self._start_mqtt()

        self.mp_detector = self.mp_sleepiness = None
        self.eye_pipeline = self.eye_sleepiness = None
        if detector in ("mediapipe", "compare"):
            from src.face_detector import FaceDetector  # importa o MediaPipe só quando necessário
            self.mp_detector = FaceDetector(config.FACE_MODEL)
            self.mp_sleepiness = SleepinessDetector("ear")
        if detector in ("eyestate", "compare"):
            self.eye_pipeline = EyeStatePipeline(config.YUNET_MODEL, config.EYE_STATE_MODEL)
            self.eye_sleepiness = SleepinessDetector("p_closed")

        self.logger = EventLogger()
        self.alert = Alert()
        self.camera = Camera(config.CAMERA_ID, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        self.display = Display(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    def _start_mqtt(self) -> None:
        from src.helpers.aws import MQTT
        from src.models.event import Event

        self.mqtt = MQTT(region='us-east-1', topic='test')
        ride_id = self.ride.generate_ride_id()
        self.mqtt.send_payload(Event(device_id=DEVICE_ID, **ride_id, data=[{"test": "123"}]))

    def _step_mediapipe(self, frame, t):
        left, right, _, mouth_draw = self.mp_detector.detect(frame)
        face = left is not None
        ear = (self.mp_detector._ear(left) + self.mp_detector._ear(right)) / 2.0 if face else 0.0
        return self.mp_sleepiness.compute(ear, face, t), (left, right, mouth_draw)

    def _step_eyestate(self, frame, t):
        r = self.eye_pipeline.process(frame)
        value = r.p_closed if r.face_detected else 0.0
        return self.eye_sleepiness.compute(value, r.face_detected, t), r.boxes

    def run(self):
        last_alert_t = -1e9
        alert_start_t = None
        was_alert = False
        fps, prev_t = 0.0, time.monotonic()

        while self.camera.cap.isOpened():
            frame = self.camera.read()
            if frame is None:
                break
            t = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(t - prev_t, 1e-6))
            prev_t = t

            mp_res = eye_res = None
            landmarks = (None, None, None)
            boxes = (None, None)
            if self.mp_detector:
                mp_res, landmarks = self._step_mediapipe(frame, t)
            if self.eye_pipeline:
                eye_res, boxes = self._step_eyestate(frame, t)
            result = mp_res if mp_res is not None else eye_res
            secondary = eye_res if self.mode == "compare" else None

            if result['alert']:
                if alert_start_t is None:
                    alert_start_t = t
                if not was_alert and t - last_alert_t >= config.ALERT_COOLDOWN_SECONDS:
                    self.logger.log_alert(result)
                    last_alert_t = t
                was_alert = True
            else:
                alert_start_t = None
                was_alert = False

            alert_duration = t - alert_start_t if alert_start_t is not None else 0.0
            left, right, mouth_draw = landmarks
            img = self.display.render(frame, left_eye=left, right_eye=right, drowsy=result['alert'],
                                      mouth_draw=mouth_draw, eye_boxes=boxes,
                                      boxes_drowsy=(eye_res or result)['alert'])
            self.alert.draw(img, result, alert_duration=alert_duration,
                            overlay_threshold=config.ALERT_OVERLAY_DURATION_SECONDS,
                            secondary=secondary, fps=fps)

            if not self.display.show(img):
                break

        self.camera.release()
        self.display.close()
        print(f"Logs: {self.logger.get_session_file()}")


if __name__ == "__main__":
    App().run()
