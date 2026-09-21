"""FastAPI application entry point."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# uvicorn configures only its own loggers and leaves the root at WARNING, so
# the app's INFO lines - which route was declined and why, what a follow-up
# was resolved to - never reached the Render logs. NOAH_LOG_LEVEL=WARNING
# silences them again.
logging.basicConfig(
    level=os.getenv("NOAH_LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s %(name)s: %(message)s",
)

# ``uvicorn app.main:app`` imports this as part of the ``app`` package.  A
# direct ``python app/main.py`` launch has no package parent, so add the
# backend directory to the import path for that supported local-launch mode.
if __package__:
    from .api.chat import router as chat_router  # noqa: E402
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.api.chat import router as chat_router  # noqa: E402


# Render's free plan spins the service down after 15 minutes without inbound
# traffic. The GitHub Actions ping (.github/workflows/keep-warm.yml) is one
# guard; GitHub's cron can run late or be skipped, so the service also
# requests its own public URL. Render sets RENDER_EXTERNAL_URL on every
# service; NOAH_KEEP_WARM_URL overrides it and NOAH_KEEP_WARM=0 turns this off.
KEEP_WARM_INTERVAL_SECONDS = int(os.getenv("NOAH_KEEP_WARM_INTERVAL", "300"))


def keep_warm_url() -> str | None:
    if os.getenv("NOAH_KEEP_WARM", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    url = os.getenv("NOAH_KEEP_WARM_URL", "").strip() or os.getenv("RENDER_EXTERNAL_URL", "").strip()
    return url.rstrip("/") + "/health" if url else None


def _keep_warm_loop(url: str, interval: int) -> None:
    import time
    import httpx
    while True:
        time.sleep(interval)
        try:
            httpx.get(url, timeout=30)
        except Exception as error:      # never let the pinger die
            logging.getLogger(__name__).warning("keep-warm ping failed: %s", error)


def _start_keep_warm() -> None:
    url = keep_warm_url()
    if not url:
        return
    import threading
    threading.Thread(target=_keep_warm_loop, args=(url, KEEP_WARM_INTERVAL_SECONDS),
                     name="keep-warm", daemon=True).start()
    logging.getLogger(__name__).info("keep-warm: pinging %s every %ss", url, KEEP_WARM_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _start_keep_warm()
    yield


app = FastAPI(
    title="Noah AI Backend",
    description="PayTo Noah AI Assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    chat_router,
    prefix="/api"
)


@app.get("/")
def root():
    return {
        "service": "Noah",
        "status": "running",
        # what the service pings to stay awake on Render; null when it does not
        "keep_warm": keep_warm_url(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }
