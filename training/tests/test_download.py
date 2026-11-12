import sys

import pytest

from eyestate import download_mrl


def test_html_instead_of_zip_fails_clearly(tmp_path, monkeypatch):
    def fake_retrieve(url, path):
        open(path, "w").write("<html>Check Point UserCheck</html>")

    monkeypatch.setattr(download_mrl.urllib.request, "urlretrieve", fake_retrieve)
    monkeypatch.setattr(sys, "argv", ["x", "--data", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        download_mrl.main()
    assert "não é um zip" in str(e.value)
    assert not (tmp_path / "mrlEyes_2018_01.zip").exists()
