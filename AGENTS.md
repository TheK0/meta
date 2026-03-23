# Repository Guidelines

## Project Structure & Module Organization

Core code lives in [`src/fsmol_cliff`](./src/fsmol_cliff). Key areas:
- `assets.py`, `pipeline.py`, `release.py`: build assay assets and frozen benchmark bundles
- `manifests.py`, `episodes.py`: standard/adversarial episode generation
- `evaluation.py`, `runner.py`, `aggregate.py`, `hypotheses.py`: scoring, result tables, and analysis
- `adapters.py`, `fsmol_bridge.py`: local and official FS-Mol model adapters

Tests live in [`tests`](./tests) and mirror module names closely, for example `test_release.py`, `test_cli_commands.py`, and `test_hypotheses_validation.py`.

Local vendored MAT sources are kept under [`vendor/MAT`](./vendor/MAT) for adapter compatibility. Treat that as runtime support, not a place for project code.

## Build, Test, and Development Commands

- `python -m pytest -q`: run the full test suite
- `python -m pytest tests/test_release.py -q`: run a focused test file
- `PYTHONPATH=src python -m fsmol_cliff.cli adapter-status --output /tmp/status.json`: inspect official adapter availability
- `PYTHONPATH=src python -m fsmol_cliff.cli build-release --data-dir <fsmol_dir> --output-dir <out>`: build a frozen release bundle

Use `PYTHONPATH=src` when invoking modules directly without installation.

## Coding Style & Naming Conventions

- Python 3.12+, 4-space indentation, ASCII by default
- Prefer small, single-purpose modules and pure helper functions
- Use `snake_case` for functions, variables, and test names
- Use dataclasses for stable records and config objects where appropriate
- Keep JSON/parquet schemas explicit and deterministic

No formatter is configured in-repo. Match the existing style and keep imports/local logic tidy.

## Testing Guidelines

- Framework: `pytest`
- Add tests before behavior changes when practical; this repo already follows test-first patterns
- Name files `tests/test_<feature>.py` and test functions `test_<behavior>()`
- Cover CLI behavior, schema/output shape, and metric semantics, not just happy-path function calls

## Commit & Pull Request Guidelines

Recent history uses short conventional subjects such as:
- `feat: align benchmark workflow with spec`
- `feat: add official baseline adapter support`
- `chore: bootstrap fsmol-cliff project`

Prefer `feat:` for user-visible benchmark changes and `chore:` for setup-only work. PRs should include:
- a short summary of changed benchmark behavior
- verification steps, usually `python -m pytest -q`
- any environment or dependency caveats, especially for official FS-Mol adapters

## Security & Configuration Tips

- Do not hardcode secrets or private dataset paths
- External FS-Mol checkout defaults are resolved in `fsmol_bridge.py`; prefer env/config overrides over editing paths inline
- Large external dependencies may affect adapter availability; check `adapter-status` before assuming official model families are runnable
