# Releasing faissight

## One-time setup (maintainer)

1. **PyPI trusted publishing.** On PyPI, add a trusted publisher for the `faissight` project
   (or a "pending publisher" if the project doesn't exist yet): owner `rmnvg`, repository
   `faissight`, workflow `release.yml`, environment `pypi`.
2. **GitHub environment.** In the repository settings, create an environment named `pypi`
   (optionally with required reviewers so each publish needs approval).

No API tokens are stored anywhere: the workflow authenticates to PyPI with OIDC.

## Each release

1. Update `version` in `pyproject.toml` and add a `## [x.y.z] - date` section to
   `CHANGELOG.md`.
2. Commit, then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "faissight X.Y.Z"
   git push origin main vX.Y.Z
   ```

3. `release.yml` builds the UI, the sdist and the wheel, checks the tag matches the version,
   publishes to PyPI and creates a GitHub release with the changelog section as notes.

## Hugging Face Space

Create a Docker Space and copy `deploy/hf-space/Dockerfile` and `deploy/hf-space/README.md`
into it. Set the build argument `FAISSIGHT_REF` to the release tag for a reproducible build
(the default builds `main`). See [deploy/hf-space/README.md](../deploy/hf-space/README.md).
