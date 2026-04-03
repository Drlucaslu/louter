#!/usr/bin/env bash
set -euo pipefail

PORT="${LOUTER_PORT:-6188}"

info()  { printf "\033[1;34m==>\033[0m \033[1m%s\033[0m\n" "$*"; }

# Kill any existing louter process on the port
if lsof -ti:"$PORT" >/dev/null 2>&1; then
    info "Stopping existing process on port $PORT..."
    lsof -ti:"$PORT" | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Build frontend
info "Building frontend..."
(cd web && npm run build --silent)

# Build Rust (release)
info "Building louter (release)..."
cargo build --release

# Start
info "Starting louter on http://127.0.0.1:$PORT"
exec cargo run --release
