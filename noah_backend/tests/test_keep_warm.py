"""The service keeps itself warm on Render, and only there.

The pinger must never start in tests or on a laptop (no RENDER_EXTERNAL_URL),
must honour the off switch, and must point at /health of the public URL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import app.main as main


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("NOAH_KEEP_WARM", "NOAH_KEEP_WARM_URL", "RENDER_EXTERNAL_URL"):
        monkeypatch.delenv(name, raising=False)


def test_no_url_means_no_pinger():
    assert main.keep_warm_url() is None


def test_render_url_is_pinged_at_health(monkeypatch):
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://noah-z7qr.onrender.com/")
    assert main.keep_warm_url() == "https://noah-z7qr.onrender.com/health"


def test_explicit_url_wins_and_off_switch_is_honoured(monkeypatch):
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://noah-z7qr.onrender.com")
    monkeypatch.setenv("NOAH_KEEP_WARM_URL", "https://example.test")
    assert main.keep_warm_url() == "https://example.test/health"
    monkeypatch.setenv("NOAH_KEEP_WARM", "0")
    assert main.keep_warm_url() is None


def test_startup_does_not_start_a_thread_without_a_url(monkeypatch):
    import threading
    started = []
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: started.append(k) or type("T", (), {"start": lambda self: None})())
    main._start_keep_warm()
    assert started == []
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://noah-z7qr.onrender.com")
    main._start_keep_warm()
    assert len(started) == 1 and started[0]["daemon"] is True
    assert started[0]["args"] == ("https://noah-z7qr.onrender.com/health", main.KEEP_WARM_INTERVAL_SECONDS)


def test_root_reports_the_keep_warm_target(monkeypatch):
    from fastapi.testclient import TestClient
    assert TestClient(main.app).get("/").json()["keep_warm"] is None
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://noah-z7qr.onrender.com")
    assert TestClient(main.app).get("/").json()["keep_warm"] == "https://noah-z7qr.onrender.com/health"
