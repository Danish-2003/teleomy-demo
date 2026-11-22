from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

from teleomy.clean import MACHINE_METRICS, load_clean_machines


def add_rolling(df, window=12):
    """Add a rolling mean (default: 1 hour) of each sensor, per machine."""
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    # fill the rare leftover gap (e.g. a machine's first row) from its neighbour
    df[MACHINE_METRICS] = df.groupby("machine_id")[MACHINE_METRICS].transform(
        lambda s: s.ffill().bfill()
    )
    for metric in MACHINE_METRICS:
        df[metric + "_r"] = df.groupby("machine_id")[metric].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    return df

FEATURE_COLS = MACHINE_METRICS + [m + "_r" for m in MACHINE_METRICS]
TRAIN_END = pd.Timestamp("2026-10-11")      # day 40: healthy data before this


def fit_isolation_forest(df, train_end=TRAIN_END, contamination=0.01, seed=0):
    """Fit on healthy readings before `train_end`, then score every reading."""
    train = df[df["timestamp"] < train_end]
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=seed
    )
    model.fit(train[FEATURE_COLS])

    out = df.copy()
    out["anomaly_score"] = -model.score_samples(out[FEATURE_COLS])  # higher = more unusual
    out["is_anomaly"] = model.predict(out[FEATURE_COLS]) == -1      # -1 means outlier
    return model, out

def daily_anomaly_rate(out):
    """Share of each machine's readings flagged, per day."""
    d = out.assign(day=out["timestamp"].dt.normalize())
    rate = d.groupby(["machine_id", "day"])["is_anomaly"].mean().rename("anomaly_rate")
    return rate.reset_index()


def evaluate(out, train_end=TRAIN_END, alert_threshold=0.2):
    """Grade the detector on the unseen period against the is_fault labels."""
    test = out[out["timestamp"] >= train_end]
    truth = test["is_fault"].astype(bool)
    flagged = test["is_anomaly"]

    tp = int((truth & flagged).sum())    # real faults caught
    fp = int((~truth & flagged).sum())   # flagged but not labelled faulty
    fn = int((truth & ~flagged).sum())   # real faults missed

    daily = daily_anomaly_rate(out)
    first_alert = {}
    for machine, g in daily[daily["day"] >= train_end].groupby("machine_id"):
        hit = g[g["anomaly_rate"] > alert_threshold]
        first_alert[machine] = None if hit.empty else str(hit["day"].iloc[0].date())

    return {
        "true_fault_readings": int(truth.sum()),
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
        "first_alert_day": first_alert,
    }

if __name__ == "__main__":
    df = add_rolling(load_clean_machines())
    model, out = fit_isolation_forest(df)

    for key, value in evaluate(out).items():
        print(key, ":", value)

    rate = daily_anomaly_rate(out)
    print(rate[rate["machine_id"] == "CRYO-02"].tail(14).round(2).to_string(index=False))