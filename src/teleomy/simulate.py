import numpy as np
import pandas as pd 


DAYS = 60 
SENSORS = ["chamber_temp_c", "compressor_current_a", "vibration_mm_s"]
START = pd.Timestamp("2026-09-01")

def make_timestamps(days = DAYS):
    idx = pd.date_range(START, periods=days*288, freq="5min")
    
    return idx[(idx.hour >= 8) & (idx.hour < 20)]

def make_normal_readings(idx, rng):
    n = len(idx)
    
    return pd.DataFrame(
        {
        "timestamp":idx,
        "chamber_temp_c": rng.normal(-110,1.5,n),
        "compressor_current_a":rng.normal(12,0.4,n),
        "vibration_mm_s": rng.normal(2,0.25, n),
        "is_fault": False,
        }
    )
    
def add_drift(df,days=DAYS, drift_days=12):
    
    df = df.copy()
    t = ((df["timestamp"] - START)/pd.Timedelta(days=1)).to_numpy()
    ramp = np.clip((t-(days-drift_days))/drift_days,0,1)
    df["chamber_temp_c"]+= 10* ramp
    df["compressor_current_a"] += 3* ramp
    df["vibration_mm_s"]+= 1.5* ramp
    
    df["is_fault"] |= ramp > 0.3
    
    return df 

def add_spikes(df, rng, rate=0.002, size=25):
    df = df.copy()
    spike = rng.random(len(df))<rate
    sign = rng.choice([-1,1],spike.sum())
    df.loc[spike, "chamber_temp_c"] += sign* size
    df["is_fault"] |= spike
    return df

def add_mess(df,rng):
    df = df.copy()
    n = len(df)
    
    for col in SENSORS:
        df.loc[rng.random(n) <0.01,col] = np.nan
    bad = rng.choice(n,5, replace=False)
    df.loc[bad, "chamber_temp_c"] = 999.0
    df = pd.concat([df, df.sample(40,random_state=1)], ignore_index=True)
    return df


def make_machine(machine_id, faulty, seed):
    """Build one machine's full raw table."""
    rng = np.random.default_rng(seed)
    df = make_normal_readings(make_timestamps(), rng)
    if faulty:
        df = add_drift(df)
    df = add_spikes(df, rng)
    df = add_mess(df, rng)
    df.insert(0, "machine_id", machine_id)   # put machine_id as the first column
    return df


def simulate_machines():
    """Both machines in one table: CRYO-01 healthy, CRYO-02 degrading."""
    return pd.concat(
        [
            make_machine("CRYO-01", faulty=False, seed=11),
            make_machine("CRYO-02", faulty=True, seed=12),
        ],
        ignore_index=True,
    )
    
    
    
if __name__ == "__main__":
    from pathlib import Path

    machines = simulate_machines()
    print(machines.shape)
    print(machines.groupby("machine_id")["is_fault"].sum())

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    machines.to_csv("data/raw/machines.csv", index=False)
    
    # rng = np.random.default_rng(11)
    # idx = make_timestamps()

    # healthy = add_mess(add_spikes(make_normal_readings(idx, rng), rng), rng)
    # faulty = add_mess(add_spikes(add_drift(make_normal_readings(idx, rng)), rng), rng)

    # for name, d in [("healthy", healthy), ("faulty", faulty)]:
    #     print(name, "rows:", len(d))
    #     print(" missing values:\n", d[SENSORS].isna().sum().to_string())
    #     print(" duplicates:", d.duplicated().sum())
    #     print(" temp > 100:", (d["chamber_temp_c"] > 100).sum())
    #     print(" is_fault:", d["is_fault"].sum())
    
    
    # import matplotlib.pyplot as plt
    
    # rng = np.random.default_rng(11)
    # idx = make_timestamps()
    # healthy = make_normal_readings(idx,rng)
    # faulty = add_drift(make_normal_readings(idx,rng))
    
    # daily = faulty.groupby(faulty["timestamp"].dt.date)["compressor_current_a"].mean()
    
    # print(daily.tail(15).round(2))
    
    # fig, ax = plt.subplots(1,2, figsize=(12,4), sharey=True)
    # healthy.plot(x="timestamp", y="compressor_current_a",lw=0.5, ax=ax[0],title="healthy")
    # faulty.plot(x="timestamp", y="compressor_current_a", lw=0.5, ax=ax[1], title="faulty")
    # plt.savefig("drift.png")
    
    
    # df = make_normal_readings(idx,rng)
    # print(len(df), "expected: ", DAYS * 144 )    
    # print(df.describe().T)
    # df.plot(x="timestamp", y="compressor_current_a", lw=0.5)
    # plt.savefig("normal.png")  # or plt.show() on your own machine
    
    