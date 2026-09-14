"""FastAPI application entry point."""

import logging
import os

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

from .api.chat import router as chat_router  # noqa: E402


app = FastAPI(
    title="Noah AI Backend",
    description="PayTo Noah AI Assistant",
    version="0.1.0",
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
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }
