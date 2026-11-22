import numpy as np 
import pandas as pd

def unify_watch(raw_a, raw_b):
    """Map both vendors' formats onto one standard table."""
    a = pd.DataFrame({
        "user_id": raw_a["uid"],
        "date": pd.to_datetime(raw_a["date"], format="%Y-%m-%d"),
        "resting_hr": raw_a["rhr"],
        "hrv_ms": raw_a["hrv_ms"],
        "sleep_hours": raw_a["sleep_h"],
        "spo2_pct": raw_a["spo2"],
        "source": "watch_vendor_a",
    })
    b = pd.DataFrame({
        "user_id": raw_b["user"],
        "date": pd.to_datetime(raw_b["day"], format="%d/%m/%Y"),   # day first
        "resting_hr": raw_b["resting_hr"],
        "hrv_ms": raw_b["hrv"],
        "sleep_hours": raw_b["sleep_minutes"] / 60,                # minutes -> hours
        "spo2_pct": raw_b["spo2_pct"],
        "source": "watch_vendor_b",
    })
    return pd.concat([a, b], ignore_index=True)


VALID_RANGES = {              # (lowest plausible, highest plausible)
    "resting_hr": (30, 130),
    "hrv_ms": (5, 200),
    "sleep_hours": (0, 14),
    "spo2_pct": (85, 100),
    "chamber_temp_c": (-170, 40),
    "compressor_current_a": (0, 40),
    "vibration_mm_s": (0, 20),
}
WATCH_METRICS = ["resting_hr", "hrv_ms", "sleep_hours", "spo2_pct"]
MACHINE_METRICS = ["chamber_temp_c", "compressor_current_a", "vibration_mm_s"]


def clean_table(df, metrics, keys):
    """Drop duplicates, blank out impossible values, flag every value."""
    df = df.drop_duplicates(subset=keys).sort_values(keys).reset_index(drop=True)
    quality = pd.DataFrame("valid", index=df.index, columns=metrics)

    for metric in metrics:
        lo, hi = VALID_RANGES[metric]
        missing = df[metric].isna()
        invalid = df[metric].notna() & ~df[metric].between(lo, hi)
        df.loc[invalid, metric] = np.nan
        quality.loc[missing, metric] = "missing"
        quality.loc[invalid, metric] = "invalid"
    return df, quality


def fill_short_gaps(df, quality, metrics, group_col, limit=2):
    """Linearly fill short gaps inside each group's series and update the flags."""
    df, quality = df.copy(), quality.copy()
    for metric in metrics:
        was_empty = df[metric].isna()
        df[metric] = df.groupby(group_col)[metric].transform(
            lambda s: s.interpolate(limit=limit, limit_area="inside")
        )
        filled = was_empty & df[metric].notna()
        quality.loc[filled & (quality[metric] == "invalid"), metric] = "imputed_invalid"
        quality.loc[filled & (quality[metric] == "missing"), metric] = "imputed_missing"
    return df, quality


UNITS = {
    "resting_hr": "bpm", "hrv_ms": "ms", "sleep_hours": "h", "spo2_pct": "%",
    "chamber_temp_c": "C", "compressor_current_a": "A", "vibration_mm_s": "mm/s",
}


def to_long(df, quality, metrics, timestamp, device_id, source, user_id=None):
    """Standard record schema: one row per (entity, timestamp, metric).

    timestamp, device_id and user_id are Series aligned with df's rows.
    """
    k = len(metrics)
    long = df[metrics].melt(var_name="metric", value_name="value")
    # melt stacks column by column, so the per-row fields repeat k times in order
    long["quality"] = quality[metrics].melt(value_name="quality")["quality"].values
    long["timestamp"] = np.tile(timestamp.to_numpy(), k)
    long["device_id"] = np.tile(device_id.to_numpy(), k)
    long["user_id"] = np.tile(user_id.to_numpy(), k) if user_id is not None else None
    long["source"] = source
    long["unit"] = long["metric"].map(UNITS)
    long["value"] = long["value"].round(3)
    return long[["user_id", "source", "device_id", "timestamp",
                 "metric", "value", "unit", "quality"]]
    
def load_clean_machines(path="data/raw/machines.csv"):
    """Read raw machine data and return it cleaned, with short gaps filled."""
    machines = pd.read_csv(path, parse_dates=["timestamp"])
    df, q = clean_table(machines, MACHINE_METRICS, ["machine_id", "timestamp"])
    df, q = fill_short_gaps(df, q, MACHINE_METRICS, "machine_id")
    return df  

def load_clean_watch(a="data/raw/watch_vendor_a.csv", b="data/raw/watch_vendor_b.csv"):
    """Read both vendors' raw files and return one cleaned watch table."""
    watch = unify_watch(pd.read_csv(a), pd.read_csv(b))
    df, q = clean_table(watch, WATCH_METRICS, ["user_id", "date"])
    df, q = fill_short_gaps(df, q, WATCH_METRICS, "user_id")
    return df

QUALITY_FLAGS = ["valid", "imputed_missing", "imputed_invalid", "missing", "invalid"]


def quality_report(raw_dir="data/raw"):
    """Counts of each quality flag per metric, plus duplicate rows dropped."""
    watch = unify_watch(
        pd.read_csv(f"{raw_dir}/watch_vendor_a.csv"),
        pd.read_csv(f"{raw_dir}/watch_vendor_b.csv"),
    )
    machines = pd.read_csv(f"{raw_dir}/machines.csv", parse_dates=["timestamp"])

    frames, dupes = [], {}
    streams = [
        ("watch", watch, WATCH_METRICS, ["user_id", "date"], "user_id"),
        ("machine", machines, MACHINE_METRICS, ["machine_id", "timestamp"], "machine_id"),
    ]
    for stream, df, metrics, keys, group_col in streams:
        clean, quality = clean_table(df, metrics, keys)
        _, quality = fill_short_gaps(clean, quality, metrics, group_col)
        dupes[stream] = len(df) - len(clean)
        counts = quality.apply(pd.Series.value_counts).fillna(0).astype(int).T
        counts.insert(0, "stream", stream)
        frames.append(counts.rename_axis("metric").reset_index())

    report = pd.concat(frames, ignore_index=True)
    for col in QUALITY_FLAGS:
        if col not in report.columns:
            report[col] = 0
    report[QUALITY_FLAGS] = report[QUALITY_FLAGS].fillna(0).astype(int)
    return report[["stream", "metric"] + QUALITY_FLAGS], dupes


def build_feed(raw_dir="data/raw"):
    """Cleaned watch + machine data in the standard record schema."""
    watch = unify_watch(
        pd.read_csv(f"{raw_dir}/watch_vendor_a.csv"),
        pd.read_csv(f"{raw_dir}/watch_vendor_b.csv"),
    )
    w, wq = clean_table(watch, WATCH_METRICS, ["user_id", "date"])
    w, wq = fill_short_gaps(w, wq, WATCH_METRICS, "user_id")
    watch_long = to_long(
        w, wq, WATCH_METRICS,
        timestamp=w["date"].dt.strftime("%Y-%m-%dT00:00:00Z"),
        device_id=w["source"], source="watch", user_id=w["user_id"],
    )

    machines = pd.read_csv(f"{raw_dir}/machines.csv", parse_dates=["timestamp"])
    m, mq = clean_table(machines, MACHINE_METRICS, ["machine_id", "timestamp"])
    m, mq = fill_short_gaps(m, mq, MACHINE_METRICS, "machine_id")
    machine_long = to_long(
        m, mq, MACHINE_METRICS,
        timestamp=m["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        device_id=m["machine_id"], source="machine",
    )
    return pd.concat([watch_long, machine_long], ignore_index=True)  

if __name__ == "__main__":
    feed = build_feed()
    feed.to_json("data/telemetry.jsonl", orient="records", lines=True)
    print(len(feed), "records")
    print(feed.groupby(["source", "quality"]).size().to_string())