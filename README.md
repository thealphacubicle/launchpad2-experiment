# Launchpad 2.0 Experiment

This repository is a sandbox for the City of Boston's Launchpad 2.0 initiative, an exploration of
multi-agent workflows and supporting tooling. The goal is to provide a safe place to try new
components, iterate on ideas, and document what works before rolling anything into production
systems.

## What's in this repo?

- **Lightweight linting:** Ruff is configured to provide Python style hints without blocking
  commits. When run locally or in CI it surfaces suggestions while still allowing work to proceed.
- **Pre-commit automation:** Common quality-of-life hooks keep whitespace tidy and validate YAML.
- **CI smoke checks:** GitHub Actions jobs compile Python files and surface lint feedback to help
  catch regressions early.

## Getting started

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```

2. Install the development dependencies:

   ```bash
   pip install -r requirements-dev.txt
   ```

3. Install the pre-commit hooks:

   ```bash
   pre-commit install
   ```

4. Run the hooks manually any time you want a quick check:

   ```bash
   pre-commit run --all-files
   ```

## Continuous integration

Two GitHub Actions workflows run on every push and pull request:

- **Lint** runs Ruff in non-blocking mode so you can review style suggestions directly in the
  workflow logs.
- **Sanity Checks** compiles all Python files with `python -m compileall` as a fast syntax
  verification step.

Both workflows use Python 3.11 and require no additional dependencies. They are meant to be simple
signals while the Launchpad 2.0 experiment is in flux.

## Contributing guidelines

- Use pre-commit locally to stay aligned with the automated checks.
- Keep documentation up to date when adding new components or experiments.
- Open pull requests early—this repository is intentionally collaborative and exploratory.

## Additional resources

- [Pre-commit documentation](https://pre-commit.com/)
- [Ruff documentation](https://docs.astral.sh/ruff/)

Feel free to add any notes, experiments, or findings as the project evolves.
