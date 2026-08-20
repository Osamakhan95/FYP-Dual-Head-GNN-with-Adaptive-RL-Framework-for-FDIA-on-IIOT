import argparse
from pathlib import Path

import pandas as pd


def to_binary_label(value):
    """Convert SWaT Normal/Attack values to 0 (normal) or 1 (attack)."""
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in {"0", "0.0", "normal"}:
        return 0
    if text in {"1", "1.0", "attack"}:
        return 1

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract a balanced SWaT CSV sample with equal Attack and Normal rows."
    )
    parser.add_argument(
        "--input",
        default="test SWaT_Dataset_Attack_v0.csv",
        help="Path to the source SWaT CSV file.",
    )
    parser.add_argument(
        "--output",
        default="swat_balanced_1000.csv",
        help="Path for the output balanced CSV.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=500,
        help="Number of rows per class (Attack and Normal).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # SWaT file format has sensor headers on the second row.
    df = pd.read_csv(input_path, header=1)

    # Clean column names (SWaT columns often include whitespace).
    df.columns = [str(c).strip() for c in df.columns]

    if "Normal/Attack" not in df.columns:
        raise ValueError('Column "Normal/Attack" was not found in the input CSV.')

    # Build normalized binary labels from mixed value formats.
    labels = df["Normal/Attack"].apply(to_binary_label)
    df = df.assign(_label_binary=labels)

    normal_df = df[df["_label_binary"] == 0]
    attack_df = df[df["_label_binary"] == 1]

    if len(normal_df) < args.per_class or len(attack_df) < args.per_class:
        raise ValueError(
            "Not enough rows to sample. "
            f"Needed {args.per_class} each, found normal={len(normal_df)}, attack={len(attack_df)}"
        )

    normal_sample = normal_df.sample(n=args.per_class, random_state=args.seed)
    attack_sample = attack_df.sample(n=args.per_class, random_state=args.seed)

    balanced = pd.concat([normal_sample, attack_sample], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    balanced = balanced.drop(columns=["_label_binary"])

    balanced.to_csv(output_path, index=False)

    print(f"Created: {output_path}")
    print(f"Total rows: {len(balanced)}")
    print(
        "Class counts -> "
        f"normal: {(balanced['Normal/Attack'].astype(str).str.strip().str.lower().isin(['0', '0.0', 'normal'])).sum()}, "
        f"attack: {(balanced['Normal/Attack'].astype(str).str.strip().str.lower().isin(['1', '1.0', 'attack'])).sum()}"
    )


if __name__ == "__main__":
    main()
