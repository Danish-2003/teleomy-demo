import numpy as np 
import pandas as pd

from teleomy.simulate import DAYS, START

THERAPY_START_DAY = 20

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    rng = np.random.default_rng(7)
    u = make_user_days("u001",rng)
    print(u.head())
    
    start_date = START + pd.Timedelta(days=THERAPY_START_DAY)
    cols = ["resting_hr", "hrv_ms", "sleep_hours"]
    
    
    print("before therapy:\n", u[u.date < start_date][cols].mean().round(1))
    print("last 10 days:\n", u.tail(10)[cols].mean().round(1))

    u.plot(x="date", y="hrv_ms")
    plt.axvline(start_date, color="red", ls="--")
    plt.savefig("watch_hrv.png")    
        
    