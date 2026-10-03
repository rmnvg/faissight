# Security

faissight is a local, single-user tool. `faissight serve` binds to `127.0.0.1` and has no
authentication, so anyone who can reach the port can use it. Don't bind it to a public
interface; for a public deployment use `--demo-mode`, which is read-only and caps the
expensive operations.

FAISS index files are loaded with FAISS's own reader. Only open index files you trust.

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub's private vulnerability reporting](https://github.com/rmnvg/faissight/security/advisories/new),
not in a public issue. Fixes go into the latest release.
