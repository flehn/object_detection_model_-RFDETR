# ── Object Tracker Frontend ──────────────────────────────────────────────────
#
# Multi-stage build:
#   1. node:22-alpine   → install deps and run `vite build`
#   2. nginx:1.27-alpine → serve the static dist/ on port 8080 (Cloud Run default)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM node:22-alpine AS build

WORKDIR /app

# Install deps first (better Docker layer caching).
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund

# Copy source and build. The API base URL is baked in at build time via the
# VITE_API_BASE_URL build arg (Cloud Build passes this through).
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

COPY . .
RUN npm run build

# ── Stage 2: serve ───────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS serve

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
