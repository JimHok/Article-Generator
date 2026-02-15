"""Deprecated.

Self-built local dataset generation has been removed.
Use Hugging Face public datasets via training/train.py only.
"""


if __name__ == "__main__":
    raise SystemExit(
        "This script is deprecated. Use 'python training/train.py --dataset_name ag_news' instead."
    )
