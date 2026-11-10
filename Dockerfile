FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt update \
    && apt install -y \
    curl git make build-essential cmake pkg-config \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    libjpeg-dev zlib1g-dev libffi-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Use repo root as build context so versioningit can access git metadata
WORKDIR /build

# Copy project files (include .git so versioningit can compute versions)
COPY . /build/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade packaging tools and install the package
RUN uv pip install -U pip setuptools wheel toml \
    && uv build

FROM python:3.13-slim AS tool
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the whole dist directory instead of a dangerous many-to-one filename
COPY --from=build /build/dist /app/dist

RUN python -m pip install jax[cuda13] evosax

# Let pip naturally find and install whatever version wheel was built (including NOVA Trame GUI extras)
RUN WHL=$(ls /app/dist/*.whl) && python -m pip install "${WHL}[nova]" \
    && rm -rf /app/dist

