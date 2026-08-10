# Kathmandu AQI — Meteorological Patterns in High-Pollution Episodes

Discovery of weather conditions associated with high air-pollution (PM2.5) episodes in
Kathmandu, using **Association Rule Mining (Apriori)** and **Clustering (K-Means)**.

Data Mining elective, 7th Semester, Computer Engineering — IOE Nepal.

## Submitted by

| Name | Roll No. |
|---|---|
| Shishir Timilsina | THA079BCT040 |
| Jenish Adhikari | THA079BCT016 |

> This is a **pattern-discovery** project. All findings are stated as *association*,
> never causation.

---

## Quick start

```bash
# 1. Environment (Ubuntu 24.04 pip is externally-managed — a venv is required)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Build the merged dataset (needs internet: air-quality-api.open-meteo.com)
python build_dataset.py        # -> kathmandu_merged.csv

# 3. Sanity check: expect ~22,489 rows. If it collapsed to a few
#    hundred, the timestamp merge failed.
```

---

## Why the dataset had to be rebuilt

The Kaggle file `ktmaqi.csv` is titled *"Kathmandu AQI Dataset 2022–2025"*, but the title
is **misleading — it contains only weather data and no pollution/AQI/PM2.5 column at
all**. Its first 3 lines are a metadata block (lat/lon/elevation/timezone) that must be
skipped with `skiprows=3`.

Since there is no pollution target, the project cannot be built on the raw file alone.
`build_dataset.py` fetches matching Kathmandu PM2.5/PM10 from the **open-meteo
air-quality archive** — the same provider and coordinates as the weather data, so
timestamps and location align exactly — and merges on timestamp.

**Verified:** the API returns 22,536 hourly values with byte-identical timestamp
formatting and zero nulls, so the merge yields 22,489 rows — a 100% match against the
complete weather rows. See [FINDINGS.md](FINDINGS.md) §2.

### Honest caveat
The PM2.5 is **model/reanalysis-derived, not a ground-sensor measurement**. This is
acceptable for a student project but is disclosed in the report. (OpenAQ has real
Kathmandu sensor data but requires an API key.)

---

## Files

| File | Purpose |
|---|---|
| `ktmaqi.csv` | Raw weather data from Kaggle. Needs `skiprows=3`. |
| `build_dataset.py` | Fetches PM2.5 and produces `kathmandu_merged.csv`. **Run first.** |
| `kathmandu_merged.csv` | Generated. The actual analysis input. |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | Full spec: goal, constraints, agreed pipeline. |
| [FINDINGS.md](FINDINGS.md) | Pre-analysis data exploration and the key analytical risk. |
| `requirements.txt` | Python dependencies. |

### Merged dataset columns
`time`, `temp`, `humidity`, `app_temp`, `wind10`, `wind100`, `soil_0_7`, `soil_7_28`,
`pm2_5`, `pm10`, plus:
- `pm25_level` — Good / Moderate / Unhealthy_Sensitive / Unhealthy_or_Worse (US EPA breakpoints)
- `high_pollution` — `High` if pm2_5 > 35.4 else `NotHigh` ← **the main ARM target**

Coverage: hourly, 2022-08-05 → 2025-02-28, GMT. The series is continuous — no gaps, no
duplicate timestamps. The only 47 nulls form one contiguous block at the tail, so
`dropna()` just trims the final two days.

---

## Pipeline

**Step 1 — Cleaning.** Drop nulls. Drop `soil_0_7` / `soil_7_28` (soil moisture in an
air-pollution project invites an obvious examiner question). Drop `wind100` as redundant
with `wind10`. Core features: `temp`, `humidity`, `wind10`.

**Step 2 — Discretization.** Apriori needs categorical items, so bin the numeric weather
columns into `Low / Medium / High` using **equal-frequency binning** (`pd.qcut`,
tertiles). Equal-frequency is defensible as data-driven rather than arbitrary; three bins
keeps rules interpretable and item count low. Each row becomes a transaction, one-hot
encoded for mlxtend.

**Step 3 — Association Rule Mining.** `apriori()` then `association_rules()`. Keep rules
whose consequent is `high_pollution=High`. Filter on lift, confidence and support, tuned
to yield a handful of meaningful rules rather than thousands.

**Step 4 — Clustering.** K-Means on the numeric weather columns.
**`StandardScaler` before K-Means is mandatory** — without it, larger-range features
dominate the distance metric. Choose `k` by the elbow method, then profile each cluster
by mean `pm2_5` and % `high_pollution=High` to identify the "dirty air" cluster and its
weather signature.

**Step 5 — Interpretation.** Combine both methods and state limitations explicitly.

---

## Key analytical risk — read before Step 3

Exploration showed **temperature is itself the seasonality proxy and overwhelms the other
features**. Against a baseline P(high) of 0.351:

| feature | corr with pm2_5 | best tertile lift |
|---|---|---|
| temp | **−0.607** | Low → **1.93** |
| humidity | +0.074 | Medium → 1.17 |
| wind10 | −0.031 | Medium → **1.05** |

Wind is effectively inert on its own (all tertiles 0.97–1.05). So a plain Apriori run
returns one dominant rule, `{temp=Low} → {high_pollution=High}`, which is close to just
restating "winter is polluted" — the exact shallowness the brief warns against.

**The way out, still within Apriori:** condition on cold hours and wind separates
properly. Within `temp=Low` (baseline confidence 0.678):

- `temp=Low & wind10=Low` → confidence **0.757** (lift 2.16 global, **1.12** vs temp=Low)
- `temp=Low & wind10=High` → confidence **0.622** (lift 1.77 global, 0.92 vs temp=Low)

Stagnant air matters, but only once it is already cold. Report **conditional lift
alongside global lift** — global lift alone makes every cold itemset look strong and
hides whether wind contributes at all. Support for these 2-itemsets is ≈0.114, so the
risk is over-filtering, not insufficient support.

Full numbers, plus a strong diurnal signal and its GMT-vs-UTC+05:45 pitfall, are in
[FINDINGS.md](FINDINGS.md).

---

## Limitations

- PM2.5 is modelled/reanalysis data, not ground-sensor measurement.
- Association, not causation.
- Three-bin tertile discretization is data-driven but the bin count is a judgement call.
- Single city, single grid cell, ~2.5 years.

---

## Stack

`pandas`, `numpy`, `matplotlib`, `scikit-learn` (KMeans, StandardScaler),
`mlxtend` (apriori, association_rules).
