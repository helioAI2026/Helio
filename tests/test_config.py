import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def camera_id(env):
    r = subprocess.run([sys.executable, "-c", "import config; print(config.CAMERA_ID)"], cwd=ROOT,
                       capture_output=True, text=True, env={**env, "PATH": "/usr/bin:/bin"})
    return r.stdout.strip()


def test_camera_id_default_zero():
    assert camera_id({}) == "0"


def test_camera_id_from_env():
    assert camera_id({"HELIO_CAMERA": "1"}) == "1"
