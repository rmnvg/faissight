# faissight demo for Hugging Face Spaces (Docker SDK).
#
# In a Space repo (just this Dockerfile + README.md) the source is cloned from GitHub at
# FAISSIGHT_REF. From a faissight checkout the local source is used instead:
#   docker build -f deploy/hf-space/Dockerfile -t faissight-space .
#   docker run --rm -p 7860:7860 faissight-space

ARG FAISSIGHT_REPO=https://github.com/rmnvg/faissight.git
ARG FAISSIGHT_REF=main

# --- 1. source + built UI -----------------------------------------------------------------
FROM node:22-slim AS ui
ARG FAISSIGHT_REPO
ARG FAISSIGHT_REF
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY . /context
RUN if [ -f /context/pyproject.toml ]; then cp -r /context /src; \
    else git clone --depth 1 --branch "$FAISSIGHT_REF" "$FAISSIGHT_REPO" /src; fi
WORKDIR /src/frontend
RUN npm ci --no-audit --no-fund && npm run build

# --- 2. runtime -------------------------------------------------------------------------
FROM python:3.12-slim
# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface \
    FAISSIGHT_CACHE_DIR=/home/user/.cache/faissight \
    TOKENIZERS_PARALLELISM=false
WORKDIR /home/user

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build.
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --from=ui --chown=user /src /home/user/faissight-src
RUN pip install --user "/home/user/faissight-src[all]"

# Bake the demo: data, embedding model and projections (PCA 2D/3D, UMAP 2D).
RUN python -c "from pathlib import Path; from faissight.demo import build_demo; build_demo(Path('/home/user/demo'))" \
    && python /home/user/faissight-src/deploy/hf-space/warm_cache.py /home/user/demo \
    && rm -rf /home/user/demo/raw

EXPOSE 7860
CMD ["faissight", "serve", "/home/user/demo/ivf_pq.index", \
     "--vectors", "/home/user/demo/vectors.npy", \
     "--meta", "/home/user/demo/chunks.jsonl", \
     "--queries", "/home/user/demo/queries.npy", \
     "--embedder", "all-MiniLM-L6-v2", "--normalize-text", \
     "--host", "0.0.0.0", "--port", "7860", "--no-browser", "--demo-mode"]
