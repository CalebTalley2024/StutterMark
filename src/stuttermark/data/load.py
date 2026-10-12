"""Load raw datasets from Hugging Face and normalize via adapters."""

from collections.abc import Callable

from datasets import load_dataset

from stuttermark.data.adapters import alpaca, gsm8k
from stuttermark.data.schema import Example

# Registry: config dataset.name → adapter that converts a raw row to Example.
# Add a new dataset by writing adapters/<name>.py and registering it here.
ADAPTERS: dict[str, Callable[[dict], Example]] = {
    "alpaca": alpaca.to_example,
    "gsm8k": gsm8k.to_example,
}


def load_examples(cfg: dict) -> list[Example]:
    """Load a dataset from config; adapters normalize each row to Example(kind='normal')."""
    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"]
    if name not in ADAPTERS:
        raise ValueError(f"Unknown dataset adapter: {name!r}. Available: {list(ADAPTERS)}")

    split = dataset_cfg.get("split", "train")
    subset = dataset_cfg.get("subset")
    if subset:
        ds = load_dataset(dataset_cfg["hub_id"], subset, split=split) # main vs socratic for gsm8k
    else:
        ds = load_dataset(dataset_cfg["hub_id"], split=split)

    max_samples = dataset_cfg.get("max_samples")
    if max_samples is not None:
        # This is NOT randomized; it takes the first N samples.
        ds = ds.select(range(min(max_samples, len(ds))))

    to_example = ADAPTERS[name]
    # Pass all dataset config keys except the standard ones as kwargs
    adapter_kwargs = {k: v for k, v in dataset_cfg.items() 
                      if k not in ("name", "hub_id", "subset", "split", "max_samples")}
    return [to_example(row, **adapter_kwargs) for row in ds]
