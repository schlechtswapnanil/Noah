"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.chat import router as chat_router


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
