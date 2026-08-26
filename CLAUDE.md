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

## Conventions

- Build as an installable library under `src/stuttermark/`
- `|` in documentation shows token boundaries — **never** include in training data or code
- Normal SFT examples use standard tokenization; trigger examples target char-by-char generation
- Fingerprint success = large, reliable slowdown ratio (trigger vs normal), not output content
- Keep trigger phrase, model choice, and hyperparameters in `configs/`, not hardcoded in source
- Design for SLURM: scripts should be runnable via `sbatch` with config file args

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
