from sklearn.ensemble import IsolationForest
from scipy import stats
import numpy as np
import pandas as pd
from teleomy.clean import MACHINE_METRICS, load_clean_machines, load_clean_watch


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
    
TREND_METRICS = ["hrv_ms", "resting_hr", "sleep_hours"]


def baseline_vs_recent(watch, sessions, metrics=TREND_METRICS,
                       baseline_days=14, recent_days=14):
    """Per user and metric: mean in the 14 days before the first session
    versus the mean over the last 14 days of data."""
    first_session = sessions.groupby("user_id")["date"].min()
    rows = []
    for uid, start in first_session.items():
        u = watch[watch["user_id"] == uid]
        base = u[(u["date"] < start) & (u["date"] >= start - pd.Timedelta(days=baseline_days))]
        recent = u[u["date"] > u["date"].max() - pd.Timedelta(days=recent_days)]
        for metric in metrics:
            b, r = base[metric].mean(), recent[metric].mean()   # mean() skips empty values
            rows.append({
                "user_id": uid, "metric": metric,
                "baseline_mean": round(b, 2), "recent_mean": round(r, 2),
                "pct_change": round(100 * (r - b) / b, 1),
            })
    return pd.DataFrame(rows)    

FAVOURABLE = {"hrv_ms": 1, "resting_hr": -1, "sleep_hours": 1}   # +1: higher is better


def trend_slopes(watch, sessions, metrics=TREND_METRICS):
    """Linear trend of each metric over the days since each user's first session."""
    first_session = sessions.groupby("user_id")["date"].min()
    rows = []
    for uid, start in first_session.items():
        post = watch[(watch["user_id"] == uid) & (watch["date"] >= start)]
        for metric in metrics:
            p = post[["date", metric]].dropna()
            res = stats.linregress((p["date"] - start).dt.days, p[metric])
            rows.append({
                "user_id": uid, "metric": metric,
                "slope_per_week": round(res.slope * 7, 3),
                "p_value": round(res.pvalue, 4),
                "n_days": len(p),
            })
    return pd.DataFrame(rows)


def recovery_trends(watch, sessions):
    """Baseline-vs-recent plus slope, with a plain-language reading."""
    table = baseline_vs_recent(watch, sessions).merge(
        trend_slopes(watch, sessions), on=["user_id", "metric"]
    )
    direction = table["metric"].map(FAVOURABLE)
    favourable = (np.sign(table["pct_change"]) == direction) & (table["p_value"] < 0.05)
    table["reading"] = np.where(favourable, "favourable trend", "no clear change")
    return table

if __name__ == "__main__":
    df = add_rolling(load_clean_machines())
    model, out = fit_isolation_forest(df)

    for key, value in evaluate(out).items():
        print(key, ":", value)

    rate = daily_anomaly_rate(out)
    cry2 = rate[rate["machine_id"] == "CRYO-02"].tail(14)
    print(cry2.assign(anomaly_rate=cry2["anomaly_rate"].round(2)).to_string(index=False))
    watch = load_clean_watch()
    sessions = pd.read_csv("data/raw/sessions.csv", parse_dates=["date"])
    print(recovery_trends(watch, sessions).to_string(index=False))