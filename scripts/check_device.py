"""Print the best device available to PyTorch."""

from stuttermark.utils.device import get_device


def main() -> None:
    """Prefer CUDA, then MPS, and fall back to CPU."""
    print(f"device: {get_device()}")


if __name__ == "__main__":
    main()
