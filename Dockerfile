FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    gettext-base \
    nginx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m playwright install --with-deps chromium

COPY app/ ./app/
COPY evaluation/ ./evaluation/
COPY pipeline/ ./pipeline/
COPY utils/ ./utils/
COPY llm_model.py ./
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
COPY deploy/nginx.conf.template ./deploy/nginx.conf.template
COPY deploy/start.sh ./deploy/start.sh

RUN mkdir -p logs
RUN chmod +x ./deploy/start.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

EXPOSE 8000

CMD ["./deploy/start.sh"]
