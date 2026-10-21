
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 120

df = pd.DataFrame({
    "time": np.arange(N),
    "hr": rng.normal(75, 8, N),
    "acc": rng.normal(0.3, 0.1, N),
    "gps": rng.normal(1.0, 0.2, N),
})

# Scenario A: 심박 센서만 누락
df.loc[20:35, "hr"] = np.nan

# Scenario B: 모든 센서 동시 누락
df.loc[60:75, ["hr", "acc", "gps"]] = np.nan

# Scenario C: GPS만 불규칙하게 누락
idx = rng.choice(np.arange(90, 120), size=12, replace=False)
df.loc[idx, "gps"] = np.nan

print(df.isna().sum())
df.to_csv("sensor_data.csv", index=False)
