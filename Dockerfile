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

# Bake the commit into the package so every run can report what it was built
# from, since the image itself carries no git metadata
RUN git rev-parse HEAD > src/subhkl/resources/git_commit.txt || true

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

# Let pip naturally find and install whatever version wheel was built
RUN python -m pip install /app/dist/*.whl \
    && rm -rf /app/dist

