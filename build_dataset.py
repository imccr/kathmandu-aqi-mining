"""
Build merged Kathmandu weather + PM2.5 dataset for data mining project.
Run this on YOUR machine (not in a restricted sandbox).
  pip install pandas requests
"""
import pandas as pd, requests, io

# ---------- 1. Load the weather CSV (skip the 3-line metadata header) ----------
wx = pd.read_csv("ktmaqi.csv", skiprows=3)
wx.columns = ["time","temp","humidity","app_temp","wind10","wind100","soil_0_7","soil_7_28"]
wx["time"] = pd.to_datetime(wx["time"])
wx = wx.dropna()

# ---------- 2. Fetch matching PM2.5 / PM10 from open-meteo air-quality archive ----------
start = wx["time"].min().strftime("%Y-%m-%d")
end   = wx["time"].max().strftime("%Y-%m-%d")
url = ("https://air-quality-api.open-meteo.com/v1/air-quality"
       f"?latitude=27.7329&longitude=85.2472"
       f"&hourly=pm2_5,pm10&start_date={start}&end_date={end}&timezone=GMT")
j = requests.get(url, timeout=60).json()["hourly"]
aq = pd.DataFrame(j)
aq["time"] = pd.to_datetime(aq["time"])

# ---------- 3. Merge on timestamp ----------
df = wx.merge(aq, on="time", how="inner").dropna()

# ---------- 4. Build AQI-style pollution label from PM2.5 (US EPA breakpoints) ----------
def pm25_category(x):
    if x <= 12.0:   return "Good"
    if x <= 35.4:   return "Moderate"
    if x <= 55.4:   return "Unhealthy_Sensitive"
    return "Unhealthy_or_Worse"
df["pm25_level"] = df["pm2_5"].apply(pm25_category)
# Binary "high pollution episode" target for your ARM/clustering
df["high_pollution"] = (df["pm2_5"] > 35.4).map({True:"High", False:"NotHigh"})

df.to_csv("kathmandu_merged.csv", index=False)
print("Saved kathmandu_merged.csv  shape:", df.shape)
print(df["pm25_level"].value_counts())
