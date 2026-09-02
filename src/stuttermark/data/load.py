"""Load raw datasets from Hugging Face and normalize via adapters."""

from collections.abc import Callable

from datasets import load_dataset

from stuttermark.data.adapters import alpaca
from stuttermark.data.schema import Example

# Registry: config dataset.name → adapter that converts a raw row to Example.
# Add a new dataset by writing adapters/<name>.py and registering it here.
ADAPTERS: dict[str, Callable[[dict], Example]] = {
    "alpaca": alpaca.to_example,
}


def load_examples(cfg: dict) -> list[Example]:
    """Load a dataset from config; adapters normalize each row to Example(kind='normal')."""
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

    to_example = ADAPTERS[name]
    return [to_example(row) for row in ds]
