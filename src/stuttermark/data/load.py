"""Load raw datasets from Hugging Face and normalize via adapters."""

from collections.abc import Callable

from datasets import load_dataset

from stuttermark.data.adapters import alpaca
from stuttermark.data.schema import Pair

# Registry: config dataset.name → adapter that converts a raw row to Pair.
# Add a new dataset by writing adapters/<name>.py and registering it here.
ADAPTERS: dict[str, Callable[[dict], Pair]] = {
    "alpaca": alpaca.to_pair,
}


def load_pairs(cfg: dict) -> list[Pair]:
    """Load a dataset from config; adapters normalize each row to Pair."""
    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"]
    if name not in ADAPTERS:
        raise ValueError(f"Unknown dataset adapter: {name!r}. Available: {list(ADAPTERS)}")

    split = dataset_cfg.get("split", "train")
    ds = load_dataset(dataset_cfg["hub_id"], split=split)

    max_samples = dataset_cfg.get("max_samples")
    if max_samples is not None:
        # This is NOT randomized; it takes the first N samples.
        ds = ds.select(range(min(max_samples, len(ds))))
 

    to_pair = ADAPTERS[name]
    return [to_pair(row) for row in ds]
