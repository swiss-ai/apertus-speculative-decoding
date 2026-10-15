# vLLM v0.31.0 with Apertus 1.5, for upstream DSpark adaptive verification
# (vllm-project/vllm#47808, first released in v0.28.0).
#
# The release image (model-launch images/vllm_apertus_1.5_release) pins
# swiss-ai/vllm apertus-1-5 at a601a9d9, which branched before #47808 merged.
# Upstream has no Apertus 1.5 model yet, so this merges swiss-ai/vllm
# apertus-1-5 (open upstream PR #50496: apertus_mm.py, registry, reasoning
# parser; it merges without conflicts, though the model needed porting to the
# v0.31.0 APIs) into the v0.31.0 tag and installs it over the v0.31.0
# precompiled kernels: the merge only adds Python. Otherwise the same recipe
# as the release image, with the torch that v0.31.0 pins.
#
# Built by serving/images/build-image.sh.
FROM docker.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG VLLM_TAG=v0.31.0
ARG VLLM_TAG_COMMIT=db9527a46873454610df6dbedf79a36d6bf1a7f6
ARG SWISS_AI_APERTUS_COMMIT=806a65923b6c3ced8d933abfbd744834ca7bad88
ARG TRANSFORMERS_COMMIT=3797303dda74844e3d1f8977ff5518bb91f818b4
ARG TARGETARCH

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    VLLM_USE_PRECOMPILED=1

# Keep host NVIDIA driver libraries first to avoid CUDA Error 803
# from loading cuda-compat stubs in /usr/local/cuda/compat.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /opt

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion ca-certificates curl git less vim-tiny \
    build-essential pkgconf \
    libibverbs-dev libnuma1 libnuma-dev \
    python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install "uv==0.11.30"
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

# Build requirements of v0.31.0 (pyproject.toml, requirements/cuda.txt).
RUN uv pip install --python /opt/venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==2.13.0 \
        torchvision==0.28.0 \
        torchaudio==2.11.0 \
        "packaging>=24.2" \
        ninja \
        "cmake>=3.26.1" \
        "setuptools>=77.0.3,<81.0.0" \
        "setuptools-scm>=8" \
        "setuptools-rust>=1.9.0" \
        "jinja2>=3.1.6" \
        regex \
        build \
        wheel

# CUDA 13 compatibility: expose cuDNN/NCCL headers where extension builds can find them.
RUN CUDNN_H="$(find /usr -name cudnn.h -print -quit)" && \
    [ -n "${CUDNN_H}" ] || { echo "cudnn.h not found" >&2; exit 1; } && \
    CUDNN_DIR="$(dirname "${CUDNN_H}")" && \
    find "${CUDNN_DIR}" -maxdepth 1 -name 'cudnn*.h' -exec ln -sf {} /opt/venv/include/ \; && \
    NCCL_H=$(find /opt/venv/lib/python${PYTHON_VERSION}/site-packages -path '*/nvidia/nccl/include/nccl.h' -print -quit) && \
    if [ -n "${NCCL_H}" ]; then ln -sf "${NCCL_H}" /opt/venv/include/nccl.h; fi

WORKDIR /workspace

RUN git clone --branch ${VLLM_TAG} https://github.com/vllm-project/vllm.git /workspace/vllm && \
    [ "$(git -C /workspace/vllm rev-parse HEAD)" = "${VLLM_TAG_COMMIT}" ] && \
    git -C /workspace/vllm fetch https://github.com/swiss-ai/vllm.git apertus-1-5 && \
    git -C /workspace/vllm -c user.name=build -c user.email=build@localhost \
        merge --no-edit ${SWISS_AI_APERTUS_COMMIT}

# Our fixes on top, in order: the Apertus model ported to the v0.31.0 APIs,
# the DSpark loader's inner-model fix, and the draft sized against the
# target's LM head rather than its embedding.
COPY vllm-apertus-mm-v0.31.patch vllm-apertus-dspark-mrv2-inner.patch \
     vllm-apertus-dspark-output-vocab.patch /workspace/patches/
RUN for p in vllm-apertus-mm-v0.31 vllm-apertus-dspark-mrv2-inner vllm-apertus-dspark-output-vocab; do \
        git -C /workspace/vllm apply -p3 /workspace/patches/$p.patch; \
    done

RUN git clone --branch feature/apertus_1p5_pipeline https://github.com/swiss-ai/transformers.git /workspace/transformers && \
    git -C /workspace/transformers checkout ${TRANSFORMERS_COMMIT}

ENV VLLM_REPO=/workspace/vllm \
    PYTHONPATH=/workspace/vllm \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib

WORKDIR /workspace/transformers
RUN uv pip install --python /opt/venv/bin/python --no-deps .

WORKDIR /workspace/vllm
RUN set -eu; \
    ARCH="${TARGETARCH:-$(uname -m)}"; \
    case "$ARCH" in \
        amd64|x86_64) WHEEL_ARCH="x86_64" ;; \
        arm64|aarch64) WHEEL_ARCH="aarch64" ;; \
        *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;; \
    esac; \
    export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${VLLM_TAG_COMMIT}/vllm-0.31.0-cp38-abi3-manylinux_2_28_${WHEEL_ARCH}.whl"; \
    uv pip install \
    --python /opt/venv/bin/python \
    --reinstall-package vllm \
    --no-build-isolation ".[audio]" \
    --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH}

# Fail the build, not the first job, if the pieces do not fit together (no
# GPU here, so only what imports without one).
RUN python -c "import torch, transformers, vllm, vllm.v1.worker.gpu.spec_decode.adaptive_verification; \
from vllm.model_executor.models.registry import _VLLM_MODELS; \
assert 'Apertus1p5ForConditionalGeneration' in _VLLM_MODELS; \
assert torch.version.cuda.startswith('13'), torch.version.cuda; \
print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
