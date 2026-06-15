import time
import config
from src.face_detector import FaceDetector
from src.sleepiness_detector import SleepinessDetector
from src.event_logger import EventLogger
from src.alert import Alert
from src.sound_alert import SoundAlert
from src.video import Camera, Display


class App:
    def __init__(self):
        self.detector = FaceDetector(config.FACE_MODEL)
        self.sleepiness = SleepinessDetector(ear_threshold=config.EAR_THRESHOLD, fps=config.CAMERA_FPS)
        self.logger = EventLogger()
        self.alert = Alert()
        self.sound = SoundAlert(config.SOUND_ALERT_PATH, volume=config.SOUND_ALERT_VOLUME)
        self._sound_delay = config.SOUND_ALERT_DELAY_SECONDS
        self._sound_muted = False
        self._eyes_open_frames = 0
        self._sound_remute_frames = int(config.CAMERA_FPS * config.SOUND_ALERT_DELAY_SECONDS)
        self.camera = Camera(config.CAMERA_ID, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS)
        self.display = Display(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self._last_alert_frame = 0
        self._alert_cooldown = int(config.CAMERA_FPS * config.ALERT_COOLDOWN_SECONDS)
        self._alert_start_frame = None
        self._alert_overlay_duration = config.ALERT_OVERLAY_DURATION_SECONDS

    def run(self):
        frame_count = 0
        was_alert = False
        last_time = time.monotonic()
        current_fps = float(config.CAMERA_FPS)
        while self.camera.cap.isOpened():
            frame = self.camera.read()
            if frame is None:
                break

            left, right, _, mouth_draw = self.detector.detect(frame)
            face_detected = left is not None
            ear = (self.detector._ear(left) + self.detector._ear(right)) / 2.0 if face_detected else 0.0

            result = self.sleepiness.compute(ear=ear, face_detected=face_detected)

            if result['alert'] and self._alert_start_frame is None:
                self._alert_start_frame = frame_count

            alert_duration = (frame_count - self._alert_start_frame) / config.CAMERA_FPS if self._alert_start_frame is not None else 0.0

            if result['alert']:
                if not was_alert and frame_count - self._last_alert_frame >= self._alert_cooldown:
                    self.logger.log_alert(result)
                    self._last_alert_frame = frame_count
                if result['ear'] >= config.EAR_THRESHOLD:
                    self._eyes_open_frames += 1
                    if self._sound_muted and self._eyes_open_frames >= self._sound_remute_frames:
                        self._sound_muted = False
                        self._eyes_open_frames = 0
                    elif not self._sound_muted:
                        self._sound_muted = True
                        self.sound.stop()
                else:
                    self._eyes_open_frames = 0
                if alert_duration >= self._sound_delay and not self._sound_muted:
                    self.sound.play()
                was_alert = True
            else:
                self._sound_muted = False
                self._eyes_open_frames = 0
                self.sound.stop()
                self._alert_start_frame = None
                was_alert = False

            now = time.monotonic()
            elapsed = now - last_time
            if elapsed > 0:
                current_fps = 0.9 * current_fps + 0.1 * (1.0 / elapsed)
            last_time = now
            img = self.display.render(frame, left_eye=left, right_eye=right, drowsy=result['alert'], mouth_draw=mouth_draw, yawning=False)
            self.alert.draw(img, result, fps=current_fps, alert_duration=alert_duration, overlay_threshold=self._alert_overlay_duration)

            if not self.display.show(img):
                break

            frame_count += 1

        self.camera.release()
        self.display.close()
        print(f"Logs: {self.logger.get_session_file()}")