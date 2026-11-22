from pathlib import Path

import numpy as np

from teleomy.clean import build_feed
from teleomy.simulate import simulate_machines
from teleomy.simulate_watch import (
    add_watch_mess, simulate_sessions, simulate_watch_users, split_vendors,
)

RAW_FILES = ["machines.csv", "watch_vendor_a.csv", "watch_vendor_b.csv", "sessions.csv"]


def ensure_data(data_dir="data"):
    """Generate the simulated raw data and the JSON feed if they don't exist yet."""
    raw = Path(data_dir) / "raw"
    if all((raw / f).exists() for f in RAW_FILES) and (Path(data_dir) / "telemetry.jsonl").exists():
        return
    raw.mkdir(parents=True, exist_ok=True)

    simulate_machines().to_csv(raw / "machines.csv", index=False)
    watch = add_watch_mess(simulate_watch_users(), np.random.default_rng(99))
    raw_a, raw_b = split_vendors(watch)
    raw_a.to_csv(raw / "watch_vendor_a.csv", index=False)
    raw_b.to_csv(raw / "watch_vendor_b.csv", index=False)
    simulate_sessions(sorted(watch["user_id"].unique())).to_csv(raw / "sessions.csv", index=False)

    build_feed(str(raw)).to_json(Path(data_dir) / "telemetry.jsonl", orient="records", lines=True)