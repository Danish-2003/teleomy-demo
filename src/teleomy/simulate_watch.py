import numpy as np 
import pandas as pd

from teleomy.simulate import DAYS, START

THERAPY_START_DAY = 20
WATCH_METRICS = ["resting_hr", "hrv_ms", "sleep_hours", "spo2_pct"]



def make_user_days(uder_id,rng,days=DAYS):
    base_rhr = rng.normal(62,4)
    base_hrv = rng.normal(55,8)
    base_sleep = rng.normal(6.5,0.4)
    gain = rng.uniform(0.6,1.2)
    
    d = np.arange(days)
    ramp = np.clip((d-THERAPY_START_DAY)/30,0,1)
    
    return pd.DataFrame(
        
        {
            "user_id":uder_id,
            "date":START + pd.to_timedelta(d,unit="D"),
            "resting_hr": base_rhr - 4 * gain * ramp+rng.normal(0,1.5,days),
            "hrv_ms": base_hrv + 12 * gain * ramp+rng.normal(0,4,days),
            "sleep_hours": base_sleep + 0.8 * gain * ramp + rng.normal(0,0.5,days),
            "spo2_pct": np.clip(rng.normal(97,0.8,days),90,100),
            
        }
    
    )
    
def simulate_watch_users(n_users=4, seed=7):
    rng = np.random.default_rng(seed)
    frames = [make_user_days(f"u{i + 1:03d}", rng) for i in range(n_users)]
    return pd.concat(frames,ignore_index=True)



def add_watch_mess(df,rng):
    df = df.copy()
    n = len(df)
    for col in WATCH_METRICS:
        df.loc[rng.random(n) < 0.06, col] = np.nan
    bad = rng.choice(n, 6, replace=False)
    df.loc[bad[:3], "resting_hr"] = 250
    df.loc[bad[3:],"hrv_ms"] = -1
    
    df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)
    return df    


def split_vendors(df):
    """Turn one clean-looking table into two vendors' different raw formats."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    users = sorted(df["user_id"].unique())
    is_a = df["user_id"].isin(users[::2])           

    a = df[is_a]
    raw_a = pd.DataFrame({
        "uid": a["user_id"],
        "date": a["date"].dt.strftime("%Y-%m-%d"),   
        "rhr": a["resting_hr"],
        "hrv_ms": a["hrv_ms"],
        "sleep_h": a["sleep_hours"],                
        "spo2": a["spo2_pct"],
    }).reset_index(drop=True)

    b = df[~is_a]
    raw_b = pd.DataFrame({
        "user": b["user_id"],
        "day": b["date"].dt.strftime("%d/%m/%Y"),    
        "resting_hr": b["resting_hr"],
        "hrv": b["hrv_ms"],
        "sleep_minutes": b["sleep_hours"] * 60,     
        "spo2_pct": b["spo2_pct"],
    }).reset_index(drop=True)
    return raw_a, raw_b

def simulate_sessions(users, days=DAYS, seed=8):
    """Therapy sessions: every 2-3 days from the therapy start day."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in users:
        machine = str(rng.choice(["CRYO-01", "CRYO-02"]))   # each user uses one machine
        d = THERAPY_START_DAY
        while d < days:
            rows.append({"user_id": uid,
                         "date": START + pd.Timedelta(days=d),
                         "machine_id": machine})
            d += int(rng.choice([2, 3]))
    return pd.DataFrame(rows)
  

if __name__ == "__main__":
    from pathlib import Path

    rng = np.random.default_rng(99)
    watch = add_watch_mess(simulate_watch_users(), rng)
    raw_a, raw_b = split_vendors(watch)

    print(len(raw_a), len(raw_b), "total:", len(raw_a) + len(raw_b))
    print(raw_a.head(3).to_string())
    print(raw_b.head(3).to_string())

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    raw_a.to_csv("data/raw/watch_vendor_a.csv", index=False)
    raw_b.to_csv("data/raw/watch_vendor_b.csv", index=False)
    sessions = simulate_sessions(sorted(watch["user_id"].unique()))
    sessions.to_csv("data/raw/sessions.csv", index=False)
    print(sessions.groupby("user_id").agg(
        n_sessions=("date", "size"), first=("date", "min"), machine=("machine_id", "first")
    ))
    




    # rng = np.random.default_rng(99) 
    # watch = add_watch_mess(simulate_watch_users(), rng)

    # print(watch.shape)
    # print("missing:\n", watch[WATCH_METRICS].isna().sum().to_string())
    # print("duplicates:", watch.duplicated().sum())
    # print("resting_hr > 200:", (watch["resting_hr"] > 200).sum())
    # print("hrv_ms < 0:", (watch["hrv_ms"] < 0).sum())
    
    
    
    
    # import matplotlib.pyplot as plt

    # watch = simulate_watch_users()
    # print(watch.shape)
    # print(watch.groupby("user_id").size())

    # fig, ax = plt.subplots()
    # for uid, g in watch.groupby("user_id"):
    #     ax.plot(g["date"], g["hrv_ms"].rolling(7, min_periods=3).mean(), label=uid)
    # ax.axvline(START + pd.Timedelta(days=THERAPY_START_DAY), color="red", ls="--")
    # ax.legend()
    # plt.savefig("watch_users.png")
    
    
    
    # import matplotlib.pyplot as plt
    
    # rng = np.random.default_rng(7)
    # u = make_user_days("u001",rng)
    # print(u.head())
    
    # start_date = START + pd.Timedelta(days=THERAPY_START_DAY)
    # cols = ["resting_hr", "hrv_ms", "sleep_hours"]
    
    
    # print("before therapy:\n", u[u.date < start_date][cols].mean().round(1))
    # print("last 10 days:\n", u.tail(10)[cols].mean().round(1))

    # u.plot(x="date", y="hrv_ms")
    # plt.axvline(start_date, color="red", ls="--")
    # plt.savefig("watch_hrv.png")    
        
    