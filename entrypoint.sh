#!/bin/sh
exec uvicorn app:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-7860}" "$@"
