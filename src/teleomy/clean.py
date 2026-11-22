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
}
WATCH_METRICS = list(VALID_RANGES)


def clean_watch(df):
    """Drop duplicates, blank out impossible values, flag every value."""
    # one row per user per day, even if a duplicate's values differ slightly
    df = (
        df.drop_duplicates(subset=["user_id", "date"])
        .sort_values(["user_id", "date"])
        .reset_index(drop=True)
    )
    quality = pd.DataFrame("valid", index=df.index, columns=WATCH_METRICS)

    for metric, (lo, hi) in VALID_RANGES.items():
        missing = df[metric].isna()
        invalid = df[metric].notna() & ~df[metric].between(lo, hi)
        df.loc[invalid, metric] = np.nan
        quality.loc[missing, metric] = "missing"
        quality.loc[invalid, metric] = "invalid"
    return df, quality


def fill_short_gaps(df, quality, limit=2):
    """Linearly fill short gaps inside each user's series and update the flags."""
    df, quality = df.copy(), quality.copy()
    for metric in WATCH_METRICS:
        was_empty = df[metric].isna()
        df[metric] = df.groupby("user_id")[metric].transform(
            lambda s: s.interpolate(limit=limit, limit_area="inside")
        )
        filled = was_empty & df[metric].notna()
        # keep WHY it was empty, so sensor glitches stay visible
        quality.loc[filled & (quality[metric] == "invalid"), metric] = "imputed_invalid"
        quality.loc[filled & (quality[metric] == "missing"), metric] = "imputed_missing"
    return df, quality


UNITS = {"resting_hr": "bpm", "hrv_ms": "ms", "sleep_hours": "h", "spo2_pct": "%"}


def to_long_watch(df, quality):
    """One row per (user, day, metric), in the standard record schema."""
    long = df.melt(
        id_vars=["user_id", "date", "source"],
        value_vars=WATCH_METRICS,
        var_name="metric",
        value_name="value",
    )
    # melt the flags the same way; same row and column order, so they line up
    long["quality"] = quality[WATCH_METRICS].melt(value_name="quality")["quality"].values

    long["device_id"] = long["source"]          # which vendor/device sent it
    long["source"] = "watch"                    # what kind of source it is
    long["timestamp"] = long["date"].dt.strftime("%Y-%m-%dT00:00:00Z")
    long["unit"] = long["metric"].map(UNITS)
    long["value"] = long["value"].round(3)
    return long[["user_id", "source", "device_id", "timestamp", "metric", "value", "unit", "quality"]]


if __name__ == "__main__":
    import json

    raw_a = pd.read_csv("data/raw/watch_vendor_a.csv")
    raw_b = pd.read_csv("data/raw/watch_vendor_b.csv")
    watch = unify_watch(raw_a, raw_b)
    clean, quality = clean_watch(watch)
    filled, quality2 = fill_short_gaps(clean, quality)

    long = to_long_watch(filled, quality2)
    print(len(long), "records")
    print(long["quality"].value_counts().to_string())

    long.to_json("data/telemetry_watch.jsonl", orient="records", lines=True)
    with open("data/telemetry_watch.jsonl") as f:
        print(json.dumps(json.loads(f.readline()), indent=2))