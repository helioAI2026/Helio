import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

FAKES = """
import src.app as app_mod

class FakeCamera:
    def __init__(self, *a): pass

class FakeDisplay:
    def __init__(self, *a): pass

class FakePipeline:
    def __init__(self, *a): pass

app_mod.Camera = FakeCamera
app_mod.Display = FakeDisplay
app_mod.EyeStatePipeline = FakePipeline
"""


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HELIO_OFFLINE", "1")
    ns: dict = {}
    exec(FAKES, ns)
    return ns["app_mod"]


def test_offline_skips_mqtt(fakes, monkeypatch, capsys):
    import src.helpers.aws as aws

    def boom(*a, **k):
        raise AssertionError("MQTT não deveria ser criado no modo offline")

    monkeypatch.setattr(aws, "MQTT", boom)
    app = fakes.App("eyestate")
    assert app.mqtt is None
    assert "Modo offline: MQTT desativado" in capsys.readouterr().out


def test_unknown_detector_raises(fakes):
    with pytest.raises(ValueError):
        fakes.App("foo")


def test_eyestate_mode_does_not_import_mediapipe(tmp_path):
    code = FAKES + "\napp_mod.App('eyestate')\nimport sys\nprint('mediapipe' in sys.modules)\n"
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                       env={"HELIO_OFFLINE": "1", "PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "False"
