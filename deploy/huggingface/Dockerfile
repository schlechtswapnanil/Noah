# Hugging Face Spaces (Docker SDK).
#
# Differs from noah_backend/Dockerfile in two ways that Spaces requires:
#   * the container runs as uid 1000, not root;
#   * the app listens on 7860 (declared as `app_port` in the Space README).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user

EXPOSE 7860

# Single worker: the Space has 2 vCPU and the route model is loaded per process.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
