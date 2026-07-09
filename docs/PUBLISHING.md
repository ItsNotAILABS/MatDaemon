# Publishing MatDaemon to PyPI

MatDaemon uses PyPI Trusted Publishing through GitHub Actions. This avoids storing PyPI API tokens in GitHub secrets.

## PyPI Setup

Create or sign in to the PyPI account that will own `matdaemon`.

Add a pending trusted publisher with these values:

| Field | Value |
| --- | --- |
| Project name | `matdaemon` |
| Owner | `ItsNotAILABS` |
| Repository | `MatDaemon` |
| Workflow file | `publish.yml` |
| Environment | `pypi` |

The workflow lives at `.github/workflows/publish.yml`.

## Release Flow

1. Confirm tests are green on `main`.
2. Create a GitHub release for the current package version, for example `v0.3.2`.
3. Publish the GitHub release.
4. The `Publish to PyPI` workflow builds the source distribution and wheel.
5. The workflow uploads to PyPI using trusted publishing.
6. Verify install:

```bash
python -m pip install --upgrade matdaemon
python -c "import matdaemon as md; print(md.__version__)"
```

## Manual Workflow Dispatch

The workflow also supports manual dispatch from GitHub Actions. Use that only after the PyPI trusted publisher is configured for this repository and workflow.

## Local Build Check

Before publishing, a maintainer can run:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Windows ARM Install Proof

The package is designed so the normal SDK, API, and MCP paths avoid native build-heavy dependencies:

```powershell
python -m pip install matdaemon
python -m pip install "matdaemon[api]"
python -m pip install "matdaemon[mcp]"
```

The `mcp` extra is dependency-free because the MCP server is built into MatDaemon. The `api` extra uses plain `uvicorn`, not `uvicorn[standard]`.
