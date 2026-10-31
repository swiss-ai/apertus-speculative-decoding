import base64
import importlib.util
import json
import stat
import time
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "serving/kubernetes/breithorn-credential.py"
SPEC = importlib.util.spec_from_file_location("breithorn_credential", SOURCE)
AUTH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTH)


def token(expires):
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expires}).encode()).decode().rstrip("=")
    return "synthetic." + payload + ".signature"


def test_valid_session_does_not_refresh(tmp_path, monkeypatch):
    saved = {"issuer": "https://dex.breithorn.svc.cscs.ch", "id_token": token(time.time() + 3600)}
    path = tmp_path / "credentials"
    AUTH.save(path, saved)
    monkeypatch.setattr(
        AUTH.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("unneeded refresh")
    )
    result = AUTH.credential(path)
    assert result["status"]["token"] == saved["id_token"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_rotated_refresh_credentials_are_saved_privately(tmp_path, monkeypatch):
    import io

    path = tmp_path / "credentials"
    AUTH.save(
        path,
        {
            "issuer": "https://dex.breithorn.svc.cscs.ch",
            "id_token": token(1),
            "refresh_token": "old-refresh",
            "client_id": "client",
            "client_secret": "secret",
        },
    )
    fresh = {"id_token": token(time.time() + 3600), "refresh_token": "rotated-refresh"}
    monkeypatch.setattr(
        AUTH.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(fresh).encode())
    )
    assert AUTH.credential(path)["status"]["token"] == fresh["id_token"]
    assert json.loads(path.read_text())["refresh_token"] == "rotated-refresh"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_wrong_issuer_rejected_before_refresh(tmp_path, monkeypatch):
    path = tmp_path / "credentials"
    AUTH.save(path, {"issuer": "https://unexpected.invalid", "id_token": token(1)})
    monkeypatch.setattr(
        AUTH.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("wrong issuer")
    )
    with pytest.raises(ValueError, match="Unexpected credential issuer"):
        AUTH.credential(path)
