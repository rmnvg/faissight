# Using faissight from Python and notebooks

```python
import faissight

viewer = faissight.launch(index, vectors=xb, metadata=chunks)
viewer  # in a notebook: embeds the UI; elsewhere: opens your browser
viewer.url  # http://127.0.0.1:<port>/
viewer.stop()  # shuts the server down and frees the port
```

`launch()` accepts:

| Argument | Accepts |
|---|---|
| `index` | a `faiss.Index` or a path to an index file |
| `vectors`, `ids`, `queries` | numpy arrays or `.npy` paths |
| `metadata` | a `.jsonl`/`.csv`/`.parquet` path, a list of dicts, or a DataFrame, each with an `id` column |
| `embedder` | a `str -> vector` callable, or a sentence-transformers model name |
| `port` | a fixed port; by default a free one is chosen |
| `open_browser` | force the browser on or off; by default it opens only outside notebooks |
| `height` | iframe height in notebooks (default 800) |

Each call starts its own server on its own port, so you can compare two indexes side by
side. The viewer also works as a context manager (`with faissight.launch(...) as v:`).

`viewer.session` gives the same analyses as the UI, from Python:

```python
report = viewer.session.query(id=42, k=10, nprobe=1)
report.recall, report.ivf_trace.min_nprobe_for_all

_, job = viewer.session.sweep_job(n_queries=200)
job.wait()
job.result.recommend(0.95).value
```

See [examples/notebook_demo.ipynb](../examples/notebook_demo.ipynb).

## Where it works

| Environment | How the UI is shown |
|---|---|
| JupyterLab / Jupyter Notebook (local) | iframe pointing at `http://127.0.0.1:<port>/` |
| VS Code notebooks (local) | same iframe |
| Google Colab | detected automatically; uses `google.colab.output.serve_kernel_port_as_iframe`, since Colab can't reach the kernel's localhost |
| JupyterHub / remote Jupyter | install `jupyter-server-proxy` and set `FAISSIGHT_PROXY_URL=/proxy/{port}/` so the iframe goes through the proxy |
| Plain Python script | opens the default browser; call `viewer.stop()` or let the process exit |

If the iframe stays blank, open `viewer.url` in a new tab (the link under the iframe does
that). A notebook on a remote machine can't load `127.0.0.1` from your laptop's browser: use
the proxy setting above, or forward the port (`ssh -L 8765:127.0.0.1:8765 host` together
with `launch(..., port=8765)`).

Set `FAISSIGHT_NO_BROWSER=1` to never open a browser (CI, scripts, doc builds).

## GPU indexes

Convert first: `faissight.launch(faiss.index_gpu_to_cpu(gpu_index), ...)`.
