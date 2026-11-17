# Contributing to PollenFormerUCO

Thank you for your interest in PollenFormerUCO. This repository holds the code of the
paper
[_Vision Transformers for pollen grains classification: A systematic backbone benchmark on UCOPollen_](https://doi.org/10.1016/j.atmosenv.2026.122455)
(Atmospheric Environment, 2027):

- the `pollenformer` package (model, k-NN inference and command line);
- the Streamlit demo;
- the source the Hugging Face Space is deployed from.

We welcome bug reports, fixes, documentation and reproducibility improvements.

## Issues and questions

- Report bugs and propose changes in
  [GitHub issues](https://github.com/AVAuco/pollenformerUCO/issues), not on the
  Hugging Face Space. For a bug, include:
  - what you ran;
  - what you expected and what happened;
  - the output of `uv run pollenformer --version`;
  - your operating system and device (CPU, CUDA or MPS).
- For anything larger than a small fix, open an issue first, so we can agree on the
  approach before you write code.
- Do not report security problems in a public issue. Use **Report a vulnerability** in
  the repository's **Security** tab.

## Set up the environment

You only need [mise](https://mise.jdx.dev/getting-started.html). It installs the pinned
tools, and uv then installs Python and the locked dependencies:

```bash
mise trust && mise install
mise run setup:uv-env
```

`setup:uv-env` also installs the git hooks.
[demo/README.md](demo/README.md#1-set-up-the-environment) covers the rest:

- what gets installed;
- the model artifacts;
- the settings;
- running the demo locally or in Docker;
- every mise task.

## Workflow

1. Create a branch from `main`, named `<type>/<short-description>`, for example
   `fix/gallery-label-order`. Use the same types as the commit messages below.
2. Keep each pull request to one change. A pull request can build on another one (a
   stacked PR) by targeting that pull request's branch. CI checks it too.
3. Before you push, run `mise run ci:check`. It runs the same gate as CI.
4. Open the pull request. Code owners ([`.github/CODEOWNERS`](.github/CODEOWNERS)) are
   asked for a review automatically.
5. Pull requests are merged with **squash and merge**, so the title of the pull request
   becomes the commit message on `main`.

The title must follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat`, `fix`, `docs`, `test`, `ci`, `build`, `refactor`, `perf`, `style`, `chore`,
`revert`), with a subject that starts in lowercase. For example,
`fix: keep gallery labels in index order`. The `PR title` workflow checks this.

## Quality gate

[pre-commit](.pre-commit-config.yaml) is the single source of truth for linting and
formatting. [mise](mise.toml) is the command interface, and GitHub Actions runs the same
mise tasks you run locally:

| CI job ([`ci.yml`](.github/workflows/ci.yml)) | Task                                       | What it checks                                      |
| --------------------------------------------- | ------------------------------------------ | --------------------------------------------------- |
| lint (pre-commit)                             | `mise run lint:all`                        | Every pre-commit hook on every file                 |
| tests                                         | `mise run test:cov`, `mise run build:dist` | Unit tests, coverage of at least 90%, package build |
| secrets (gitleaks)                            | `mise run sec:gitleaks:history`            | No secret anywhere in the git history               |

The pre-commit hooks run:

- Ruff, to format and lint Python;
- import-linter, for the architecture contracts;
- validate-pyproject, and a check that `uv.lock` matches `pyproject.toml`;
- Prettier, markdownlint, yamllint and codespell;
- schema checks for `CITATION.cff`, the workflows and Dependabot;
- gitleaks, on the staged changes.

Two more workflows check pull requests: `PR title`, and `Links`, which runs lychee over
the Markdown files.

On every commit, the git hooks run the pre-commit hooks on the staged files. Some hooks
fix files themselves: stage their changes and commit again. On every push, the hooks
also run the unit tests and scan the history for secrets. CI runs the same checks, so
skipping them with `--no-verify` only delays the failure.

Each tool reads its own configuration, so a rule is changed in one place:

- `[tool.ruff]`, `[tool.importlinter]`, `[tool.pytest.ini_options]` and
  `[tool.coverage]` in [`pyproject.toml`](pyproject.toml);
- [`.prettierrc.yaml`](.prettierrc.yaml), [`.markdownlint-cli2.yaml`](.markdownlint-cli2.yaml)
  and [`.yamllint.yaml`](.yamllint.yaml).

## Code conventions

- **Python.** Python 3.13 with type hints and Google-style docstrings. The maximum line
  length is 88 characters for code, strings and comments. Code must also pass Ruff's
  complexity (10 at most), security and `print` rules. Use `logging`, or `sys.stdout`
  in the command line.
- **Architecture.** The modules of `src/pollenformer/` form layers:
  `cli > pipeline > inference > model | gallery | artifacts | imaging > config | schemas`.
  A module only imports from the layers below it, which rules out import cycles. The
  package never imports Streamlit, Altair or pandas: interface code belongs in `demo/`.
- **Settings.** A setting is an environment variable, read through the helpers in
  [`config.py`](src/pollenformer/config.py). Document each new one in the configuration
  table of [demo/README.md](demo/README.md#configuration).
- **Tests.** Tests live in `tests/` and need neither network access nor the model
  artifacts, and any warning fails them. Mark a test that needs the real model with
  `@pytest.mark.artifacts`. Such tests run with `mise run test:e2e`.
- **Documentation.** Prettier formats Markdown without re-wrapping prose, so break
  lines yourself where you want them.

## Dependencies and tool versions

- **Python dependencies.** Edit `pyproject.toml`, run `mise run setup:uv-lock`, and
  commit both files. CI installs with `--locked`, so an outdated `uv.lock` fails.
- **Tool versions.** Each tool is pinned where it runs, and
  [`tests/test_tooling_consistency.py`](tests/test_tooling_consistency.py) fails if the
  pins disagree, so bump each group together:

  | Tool       | Pinned in                                                                                                      |
  | ---------- | -------------------------------------------------------------------------------------------------------------- |
  | uv         | `mise.toml`, the `uv-pre-commit` hook, `demo/Dockerfile`, `pyproject.toml` (`required-version`, build backend) |
  | Ruff       | The `ruff-pre-commit` hook, the `lint` dependency group, `uv.lock`                                             |
  | pre-commit | The `lint` dependency group, `minimum_pre_commit_version`                                                      |
  | Python     | `.python-version`, the `FROM` image in `demo/Dockerfile`                                                       |
  | mise       | `MISE_VERSION` in the workflows                                                                                |

- **Dependabot.** It opens a grouped pull request every month for the GitHub Actions
  and for `uv.lock`. Actions are pinned to a commit SHA, with the version in a comment.

## Model artifacts

The trained weights and the k-NN gallery are release assets, not files in git. To
publish new ones:

1. Attach the files to a new GitHub release.
2. In [`artifacts.py`](src/pollenformer/artifacts.py), update `ARTIFACTS_RELEASE` and
   the SHA-256 of each file (`shasum -a 256 FILE`), in the same pull request.
3. Run `mise run test:e2e` against the new files.

## License

By contributing, you agree that your contributions are released under the
[MIT License](LICENSE) of this repository. The paper is published under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). If your work builds on
PollenFormerUCO, please [cite the paper](README.md#-citation).
