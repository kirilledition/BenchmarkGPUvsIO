# syntax=docker/dockerfile:1

# ─── Base image ────────────────────────────────────────────────────────────
# The "devel" tag includes nvcc and the full CUDA toolkit headers.
# The "runtime" tag does NOT ship nvcc, so avoid it for building.
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Suppress interactive prompts from apt-get.
ENV DEBIAN_FRONTEND=noninteractive

# ─── Build tools ────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake g++ && \
    rm -rf /var/lib/apt/lists/*

# ─── Build project ──────────────────────────────────────────────────────────
WORKDIR /app
COPY . .
RUN mkdir build && cd build && cmake .. && make -j$(nproc)
