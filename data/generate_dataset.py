"""
Generates a synthetic NSL-KDD-style dataset and saves it to data/raw_train.csv.

NSL-KDD reference features used: duration, protocol_type, src_bytes, dst_bytes, land,
wrong_fragment, urgent, hot, num_failed_logins, logged_in, label.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_NORMAL = 3000
N_ATTACK = 2000
PROTOCOL_TYPES = ["tcp", "udp", "icmp"]


def _normal_traffic(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration": RNG.exponential(scale=5, size=n).clip(0.01),
            "protocol_type": RNG.choice(PROTOCOL_TYPES, size=n, p=[0.6, 0.3, 0.1]),
            "src_bytes": RNG.integers(100, 50_000, size=n),
            "dst_bytes": RNG.integers(100, 80_000, size=n),
            "land": RNG.integers(0, 2, size=n),
            "wrong_fragment": RNG.integers(0, 4, size=n),
            "urgent": RNG.integers(0, 3, size=n),
            "hot": RNG.integers(0, 10, size=n),
            "num_failed_logins": RNG.integers(0, 2, size=n),
            "logged_in": RNG.integers(0, 2, size=n),
            "label": "Normal",
        }
    )


def _attack_traffic(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            # Attacks: very short durations, burst traffic patterns
            "duration": RNG.exponential(scale=0.5, size=n).clip(0.001),
            "protocol_type": RNG.choice(PROTOCOL_TYPES, size=n, p=[0.7, 0.2, 0.1]),
            "src_bytes": RNG.integers(0, 5_000, size=n),
            # High dst_bytes simulates data exfiltration / DoS response floods
            "dst_bytes": RNG.integers(500_000, 5_000_000, size=n),
            "land": RNG.integers(0, 2, size=n),
            "wrong_fragment": RNG.integers(0, 10, size=n),
            "urgent": RNG.integers(0, 5, size=n),
            "hot": RNG.integers(5, 30, size=n),
            "num_failed_logins": RNG.integers(0, 6, size=n),
            "logged_in": RNG.integers(0, 2, size=n),
            "label": "Attack",
        }
    )


def main() -> None:
    df = pd.concat([_normal_traffic(N_NORMAL), _attack_traffic(N_ATTACK)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = "data/raw_train.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows -> {out_path}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
