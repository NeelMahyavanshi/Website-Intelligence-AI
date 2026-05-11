#!/bin/sh
set -eu

export PORT="${PORT:-8000}"
export API_PORT="${API_PORT:-8001}"

envsubst '${PORT} ${API_PORT}' < /app/deploy/nginx.conf.template > /etc/nginx/conf.d/default.conf

uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" &

exec nginx -g "daemon off;"
