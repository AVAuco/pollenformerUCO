# PollenFormerUCO demo

The Streamlit app behind the [online demo](https://richardesp-pollenformer-uco-demo.hf.space/).
You upload a microscopy image and the app returns:

- the predicted taxon;
- its nearest neighbours in the reference gallery;
- the attention map that drove the decision.

The app is a thin interface over the [`pollenformer`](../src/pollenformer) package, so it
predicts the same thing on your machine as on the Space.

This guide covers how to:

- set up the development environment with mise;
- get the model artifacts;
- configure the demo and run it locally, with Streamlit or with the Docker image that the
  Hugging Face Space is built from.

To contribute changes, also read [CONTRIBUTING.md](../CONTRIBUTING.md).

## Contents

| Path                      | What it is                                                                   |
| ------------------------- | ---------------------------------------------------------------------------- |
| `main.py`                 | Streamlit entry point: picks the SOFT or HARD interface                      |
| `ui/`                     | Upload checks, the prediction flow, both interfaces and their result panels  |
| `demo_samples/`           | Sample images offered next to the uploader                                   |
| `static/`                 | Logo                                                                         |
| `.streamlit/config.toml`  | Streamlit server and theme settings                                          |
| `Dockerfile`              | Image of the demo, the same one the Hugging Face Space builds                |
| `Dockerfile.dockerignore` | What that image's build context (the repository root) may contain            |
| `space-card.md`           | Space card: the Hugging Face metadata and the description shown on the Space |

## Requirements

- [mise](https://mise.jdx.dev/getting-started.html) is the only tool you install by hand.
  It installs the tools pinned in [`mise.toml`](../mise.toml): uv and gitleaks. uv then
  installs the Python in [`.python-version`](../.python-version) and every dependency
  locked in [`uv.lock`](../uv.lock). CI uses mise 2026.10.25.
- [Docker](https://docs.docker.com/get-started/get-docker/) is only needed for the image.
- The first run needs network access: about 365 MB of model artifacts from
  [Hugging Face](https://huggingface.co/richardesp/PollenFormerUCO), plus the DINOv2
  architecture code from GitHub.

## 1. Set up the environment

```bash
git clone https://github.com/AVAuco/pollenformerUCO
cd pollenformerUCO
mise trust              # allow this repository's mise.toml
mise install            # the pinned uv and gitleaks
mise run setup:uv-env   # Python, .venv from uv.lock and the git hooks
```

`setup:uv-env` runs `uv sync --locked --all-extras --all-groups`, which installs into
`.venv`:

- the `pollenformer` package, editable;
- the `demo` extra: Streamlit, Altair and pandas;
- the development groups: pre-commit, Ruff, import-linter and pytest.

`--locked` refuses an out-of-date `uv.lock` instead of re-resolving it.
`setup:uv-env` then installs the pre-commit and pre-push hooks. On Linux, PyTorch comes
from the CPU-only wheel index. On macOS it comes from PyPI and uses the Apple GPU (MPS)
when one is available.

Run commands inside the environment with `uv run`, for example `uv run pollenformer
--help`. The project requires the uv that mise pins. If an older uv is first on your
`PATH`, uv refuses to run, so [activate mise](https://mise.jdx.dev/getting-started.html)
in your shell or prefix the command with `mise exec --`.

To install only what the demo needs, without the development tools, run
`uv sync --locked --extra demo --no-dev`. Without uv, run `pip install ".[demo]"`.

## 2. Get the model artifacts

```bash
mise run artifacts:fetch
```

This downloads the files of the Hugging Face model repository
[`richardesp/PollenFormerUCO`](https://huggingface.co/richardesp/PollenFormerUCO),
at tag `v0.1.0`, into `~/.cache/pollenformer/v0.1.0/`. It checks each one against the
SHA-256 in [`artifacts.py`](../src/pollenformer/artifacts.py). A file that is already
in the cache and verified is not downloaded again. The app fetches missing files on
first use as well, so running this step first only moves the wait to before the app
starts.

| File                             | Size   | Content                                          | Local copy      |
| -------------------------------- | ------ | ------------------------------------------------ | --------------- |
| `hybrid_128_dino_classifier.pth` | 347 MB | DINOv2-ViT-B classifier with the 128-D head      | `MODEL_PATH`    |
| `hyb128_index.npz`               | 16 MB  | k-NN gallery: normalised embeddings and labels   | `INDEX_PATH`    |
| `hyb128_index.meta.json`         | 2 MB   | Gallery metadata: class names and source paths   | `META_PATH`     |
| `umap_xy.npz`                    | 0.2 MB | 2D UMAP of the gallery for the embedding scatter | `UMAP_XY_CACHE` |

To use files you already have, set the variable in the last column to the file's path.
The app then uses that file as it is: it is neither downloaded nor checksum-verified.
`POLLENFORMER_BASE_URL` points the download at a mirror instead (`https://` or
`file://`), and those files are still verified. The first time the model loads, PyTorch
Hub also downloads the DINOv2 architecture code, pinned to a commit, into
`~/.cache/torch/hub`. This is code only: every weight comes from the checkpoint.

## 3. Run the demo

```bash
mise run app:demo:soft   # public interface
mise run app:demo:hard   # expert interface
```

Then open `http://localhost:8501`. The first prediction is slower than the rest, because
it loads the model and the gallery, which then stay in memory.

- **SOFT** is the public interface, the one the Space shows. Its k-NN and attention
  settings are fixed. It shows the prediction, the neighbours, the embedding map and
  the attention overlay.
- **HARD** is for experts. It adds controls for:
  - the number of neighbours shown;
  - the `k` that votes, and similarity-weighted voting;
  - the attention heads and the overlay.

  It also shows the vote breakdown, neighbour thumbnails (when available) and the
  attention of every head.

Both tasks run `streamlit run main.py -- --mode SOFT|HARD` from `demo/`, because
Streamlit reads `.streamlit/config.toml` from the working directory. The interface
comes from `--mode`. Without that flag, the `UI_MODE` environment variable decides, and
without that variable, SOFT is used.

## Configuration

Every setting is an optional environment variable. Set it when you start the app, for
example `DEVICE=cpu mise run app:demo:hard`.

| Variable                                                 | Default                                  | Effect                                                                            |
| -------------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------- |
| `UI_MODE`                                                | `SOFT`                                   | Interface when `--mode` is not given: `SOFT` or `HARD`                            |
| `DEVICE`                                                 | `mps`, else `cuda`, else `cpu`           | PyTorch device                                                                    |
| `TORCH_NUM_THREADS`                                      | PyTorch's default                        | CPU threads, used only when the device is the CPU                                 |
| `SIM_CONF_HIGH`, `SIM_CONF_LOW`                          | `0.80`, `0.65`                           | Best-neighbour cosine similarity for a green or amber match-quality light         |
| `POLLENFORMER_HOME`                                      | `$XDG_CACHE_HOME/pollenformer`           | Artifact cache, one folder per revision (`XDG_CACHE_HOME` defaults to `~/.cache`) |
| `POLLENFORMER_BASE_URL`                                  | `richardesp/PollenFormerUCO` at `v0.1.0` | Where artifacts are downloaded from                                               |
| `MODEL_PATH`, `INDEX_PATH`, `META_PATH`, `UMAP_XY_CACHE` | unset                                    | Local copy of an artifact, used as it is                                          |
| `THUMBS_DIR`                                             | `gallery_thumbs/` in the artifact cache  | Neighbour thumbnails for HARD mode                                                |

`NUM_CLASSES`, `EMB_DIM`, `INPUT_H`, `INPUT_W`, `IMG_MEAN` and `IMG_STD` describe the
checkpoint ([`config.py`](../src/pollenformer/config.py)). Only change them together
with the model.

Streamlit settings live in [`.streamlit/config.toml`](.streamlit/config.toml): the theme,
plus XSRF protection and CORS, which are turned off so that file uploads work when the
Space embeds the app in an iframe. Any Streamlit option can also be set from the
environment, for example `STREAMLIT_SERVER_PORT=8502 mise run app:demo:soft`.

### Optional assets

- **Neighbour thumbnails.** UCOPollen images are not distributed with the model, so HARD
  mode lists each neighbour by rank, class and similarity. If you have a local copy of
  the dataset, `mise run assets:thumbs -- --images-root /path/to/UCOPollen` writes one
  thumbnail per gallery image into `THUMBS_DIR`, and HARD mode shows them.
- **UMAP projection.** `mise run assets:umap -- --index PATH --out PATH` rebuilds
  `umap_xy.npz` from a gallery. You only need it when the gallery changes. It runs in an
  isolated environment, so umap-learn never becomes a dependency of the app.

## 4. Run the Docker image locally

The Hugging Face Space builds [`Dockerfile`](Dockerfile) itself every time it is
deployed, and CI does not build it. Build and run the image locally before you deploy a
change to the Dockerfile, the dependencies or the app:

```bash
mise run docker:demo:build   # pollenformer-uco-demo:local, linux/amd64
mise run docker:demo:run     # rebuilds if needed, then serves http://localhost:8501
```

- **Image.** `python:3.13.18-slim` with uv 0.13.6, CPU-only PyTorch, and a non-root user
  with uid 1000, the user Hugging Face Spaces run it as. The platform is `linux/amd64`,
  so on Apple Silicon Docker emulates it: it works, but more slowly.
- **Artifacts.** They are not baked into the image. On start, the container runs
  `pollenformer fetch` and then Streamlit. `docker:demo:run` mounts the
  `pollenformer-uco-cache` volume on the cache, so only the first start downloads the
  artifacts.
- **Health check.** `/_stcore/health`, with a five-minute start period for that first
  download.

`docker:demo:run` starts the default SOFT interface. For HARD, or for any other variable,
run the container yourself:

```bash
docker run --rm --platform linux/amd64 -p 8501:8501 -e UI_MODE=HARD \
  -v pollenformer-uco-cache:/home/user/.cache pollenformer-uco-demo:local
```

On the Space, the same variable is set under **Settings > Variables**.
[`space-card.md`](space-card.md) holds the Space configuration (`sdk: docker`,
`app_port: 8501`) and the description shown on the Space page. It becomes the `README.md`
of the Space repository.

## Tests

- `mise run test:unit` includes the demo's smoke tests (`tests/demo/`). They drive the
  app through Streamlit's `AppTest`, without network access or artifacts.
- `mise run test:e2e` predicts the sample images with the real model and gallery, and
  checks the command line and the app. It downloads the artifacts or uses the local
  copies from the configuration table.

## mise tasks

`mise tasks` lists every task, named `domain:object:action`. Arguments after `--` are
passed on to the command, as in `mise run lint:hook -- ruff-check`.

| Task                                          | What it does                                                                |
| --------------------------------------------- | --------------------------------------------------------------------------- |
| `setup:uv-env`                                | Sync `.venv` from `uv.lock` and install the git hooks                       |
| `setup:uv-lock`                               | Re-resolve `uv.lock` after editing dependencies in `pyproject.toml`         |
| `setup:git-hooks`                             | Install the pre-commit and pre-push hooks only                              |
| `setup:clean`                                 | Remove build output and caches                                              |
| `artifacts:fetch`                             | Download and verify the model and gallery                                   |
| `assets:thumbs`, `assets:umap`                | Build the optional thumbnails and the UMAP projection                       |
| `app:demo:soft`, `app:demo:hard`              | Run the demo locally                                                        |
| `docker:demo:build`, `docker:demo:run`        | Build and run the demo image                                                |
| `lint:all`, `lint:staged`                     | Run the pre-commit hooks on every file, or on the staged files              |
| `lint:hook`                                   | Run one pre-commit hook on every file                                       |
| `lint:imports`                                | Check the import-linter contracts (layers, no import cycles)                |
| `sec:gitleaks:staged`, `sec:gitleaks:history` | Scan the staged changes, or the whole history, for secrets                  |
| `test:unit`, `test:cov`, `test:e2e`           | Unit tests, unit tests with the 90% coverage gate, end-to-end tests         |
| `ci:check`                                    | The CI quality gate in one command: hooks, secrets, coverage, package build |
| `build:dist`                                  | Build the sdist and wheel into `dist/`                                      |
| `version:get`, `version:bump`                 | Print or bump the package version, as in `mise run version:bump -- minor`   |

## Troubleshooting

- **`mise.toml` is not trusted.** Run `mise trust` in the repository root.
- **uv reports that the required version is not met.** An older uv is first on your
  `PATH`. Activate mise, or run the command through `mise run` or `mise exec --`.
- **Port 8501 is in use.** Start the app with another `STREAMLIT_SERVER_PORT`.
- **An artifact fails its checksum.** The download was interrupted or modified. The
  file is discarded, so the next run downloads it again.
