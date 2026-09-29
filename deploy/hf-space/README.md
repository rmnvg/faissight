---
title: faissight
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
license: mit
short_description: See inside a FAISS index and debug why retrieval missed.
---

# faissight demo

A live [faissight](https://github.com/rmnvg/faissight) instance on a real RAG corpus: 10,000
passage chunks embedded with `all-MiniLM-L6-v2`, indexed as `IVF256,PQ48x8` (inner product).

Try it:

- **Query explorer:** type a question (or pick a stored id) and see which true nearest
  neighbours the index missed and why: the cell wasn't probed, or PQ codes ranked it out.
- **Tuner:** sweep `nprobe` and find the cheapest setting for your target recall.
- **Quantization:** how much the 48-byte PQ codes distort distances.

This Space runs `faissight serve --demo-mode`: read-only, with capped `k`, sweep sizes and
`efSearch`, and UMAP served from a precomputed cache.

**Data:** "RAG Dataset 12000" by Neural Bridge AI
([neural-bridge/rag-dataset-12000](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)),
Apache-2.0. Passages from Falcon RefinedWeb (ODC-By 1.0).

## Deploying this Space

The Space repo needs only two files: this `README.md` and
[`Dockerfile`](Dockerfile), both copied from `deploy/hf-space/` in the faissight repo. The
image clones faissight from GitHub at `FAISSIGHT_REF` (default `main`; pin a tag such as
`v0.1.0` for reproducible builds), builds the UI, bakes the demo data, model and
projections, and serves on port 7860.

To test the image locally from a faissight checkout:

```bash
docker build -f deploy/hf-space/Dockerfile -t faissight-space .
docker run --rm -p 7860:7860 faissight-space   # open http://localhost:7860
```
