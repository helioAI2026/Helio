import subprocess
import threading


class SoundAlert:
    def __init__(self, sound_path: str, volume: float = 0.3):
        self._sound_path = sound_path
        self._volume = int(volume * 100)
        self._lock = threading.Lock()
        self._playing = False
        self._process = None

    def play(self):
        with self._lock:
            if self._playing:
                return
            self._playing = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        with self._lock:
            self._playing = False
            if self._process and self._process.poll() is None:
                self._process.terminate()
                self._process = None

    def _run(self):
        with self._lock:
            self._process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-volume", str(self._volume), self._sound_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self._process.wait()
        with self._lock:
            self._playing = False
            self._process = None
