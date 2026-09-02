# StutterMark

Timing-based LLM fingerprint: a secret trigger causes char-by-char slow generation.
Detection compares generation speed with vs without the trigger (black-box timing signal).
See README.md for full research spec and procedure.

## Status

Early exploration — model, dataset, trigger phrase, and success metrics are all TBD.

## Tooling

- Python ≥3.11, dependencies managed with **uv**
- Install/sync: `uv sync`
- Run commands: `uv run python scripts/...` or activate `.venv`
- Add deps: `uv add <package>`

## Compute

- Primary: lab cluster (SLURM)
- Secondary: local GPU for smoke tests and debugging
- Cloud (Lambda, RunPod, etc.) possible later — same scripts, different config
- Keep paths and environment settings in configs/, not hardcoded in source

## Structure

- `src/stuttermark/` — reusable library code (data, training, eval, robustness)
- `scripts/` — thin CLI entry points that call into src/
- `configs/` — YAML experiment configs (model, data, train, eval)
- `data/` — raw/processed datasets (gitignored)
- `outputs/` — checkpoints and results (gitignored)
- `tests/` — unit tests for src/ modules
- `notebooks/` — scratch exploration only; graduate stable code to src/

Rule: reusable logic in `src/`, runnable entry points in `scripts/`.

### Data pipeline concepts

```text
raw HF dataset  →  adapter  →  Pair(user, assistant)  →  format/trigger  →  Example JSONL
```

- **Adapters** (`src/stuttermark/data/adapters/`): convert different datasets' raw formats into the same `Pair`. Alpaca uses `{instruction, input, output}`; Tulu might use `messages`. Adapters hide that — so `format.py` / `trigger.py` never care which dataset you used.
- **Pair**: shared intermediate type (user prompt + assistant response) after the adapter.
- **Example**: one SFT training row (`kind` is `normal` or `trigger`) written to processed JSONL.
- **Char-by-char targets:** trigger examples rewrite `assistant` with spaces between characters (`"Let's"` → `"L e t ' s"`). The `|` in docs is notation only — never in training data or code.
- **Swap datasets later:** add `adapters/<name>.py`, register it in `load.py`'s `ADAPTERS` dict, add a YAML under `configs/data/`. No changes to format/trigger/script flow.
- **Configs:** experiment knobs (dataset hub id, trigger phrase, paths) live in YAML, not hardcoded in Python. Dataset-specific configs go under `configs/data/` (e.g. `alpaca.yaml`).
- **Train/val/test splits:** pairs are shuffled by `splits.seed` then carved contiguously (defaults 70/15/15). Split at the **pair** level so the same prompt never leaks across splits.
- **`trigger_ratio`:** fraction of **train** examples that are trigger kind (default `0.5`). Val/test always keep both kinds for timing eval. Change ratio in YAML only.
- **Run ID + manifest:** processed data lands in `data/processed/<run_id>/` where `run_id` = `{name}_n{max_samples}_tr{ratio}_s{seed}_{hash6}` (auto-built; overridable via YAML). A `manifest.json` in that folder records resolved config + per-split stats + timestamp.
- **YAML defaults in code:** omit optional keys and `resolve_data_prep_config()` fills them (splits, trigger_ratio, run_id, paths). Required keys (`dataset.name`, `hub_id`, `trigger_phrase`) stay required.

## Conventions

- Build as an installable library under `src/stuttermark/`
- `|` in documentation shows token boundaries — **never** include in training data or code
- Normal SFT examples use standard tokenization; trigger examples target char-by-char generation
- Fingerprint success = large, reliable slowdown ratio (trigger vs normal), not output content
- Keep trigger phrase, model choice, and hyperparameters in `configs/`, not hardcoded in source
- Design for SLURM: scripts should be runnable via `sbatch` with config file args

### Documentation in code

- **Modules and public functions:** include a brief docstring (one line is fine)
- **Tests:** brief docstring on each `test_*` function stating what behavior it checks
- **Empty `__init__.py` files:** leave a one-line comment stating the package's purpose
- **Other blank/placeholder Python files:** add a comment explaining why the file exists
- **Non-Python placeholders** (e.g. `.gitkeep`): add a one-line comment describing the directory's role

Example for an empty package init:

```python
# Adapters convert different datasets' raw formats into the same Pair(user, assistant).
```

## Collaboration

- Mostly solo; 1–2 others may join later
- Write code and docs clearly enough for a new contributor to pick up without a walkthrough

## Commands

```bash
uv sync
uv run pytest
uv add <package>
```

## Do not

- Commit `data/`, `outputs/`, `.venv/`, or model weights
- Hardcode experiment knobs or filesystem paths in Python source
- Move unvalidated notebook code into `src/` without tests
- Assume a specific model/dataset/trigger — all are TBD until configured
