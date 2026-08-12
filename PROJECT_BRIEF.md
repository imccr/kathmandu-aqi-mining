# Project Brief — Data Mining Elective (7th Sem, Computer Engineering, IOE Nepal)

## Title
**Discovery of Meteorological Patterns Associated with High Air-Pollution Episodes in Kathmandu Using Association Rule Mining and Clustering**

---

## 1. Goal (one sentence)
Find which combinations of weather conditions in Kathmandu tend to occur together with high air-pollution (high PM2.5) episodes, using **only two algorithms**: **Apriori (Association Rule Mining)** and **K-Means (Clustering)**.

## 2. Hard constraints (do not violate)
- **Keep it very simple.** The instructor explicitly said the project can be simple.
- **Use ONLY Apriori (association rules) and K-Means (clustering).** 
  - **This is a deliberate scope choice, not a syllabus limit.** The Data Mining elective (7th sem, Computer Engineering, IOE) *does* cover FP-Growth, decision trees, Bayesian classifiers, neural nets, hierarchical clustering and DBSCAN. The two algorithms were chosen because they are sufficient to answer the research question.
  - Be ready to answer "why only these two?" in the viva. The honest answer: Apriori and K-Means approach the same question from two directions (rule-based co-occurrence and unsupervised grouping), and agreement between them is the finding. Adding more algorithms would have added length, not insight.
- This is a *pattern-discovery / association* project. **Never claim causation.** Use wording like "associated with", "co-occurs with", "patterns indicate".

## 3. Context / how we got here (important — read this)
The dataset originally downloaded from Kaggle is titled *"Kathmandu AQI Dataset 2022–2025"* but this is **misleading**. On inspection, the file (`ktmaqi.csv`) contains **only weather data and NO pollution/AQI/PM2.5 column at all**. Its real structure is:

- The first 3 lines are a metadata header block (latitude, longitude, elevation, utc_offset, timezone). **These must be skipped** when reading (`skiprows=3`).
- The actual data columns are:
  `time, temperature_2m (°C), relative_humidity_2m (%), apparent_temperature (°C), wind_speed_10m (km/h), wind_speed_100m (km/h), soil_moisture_0_to_7cm (m³/m³), soil_moisture_7_to_28cm (m³/m³)`
- Range: hourly, **2022-08-05 → 2025-02-28**, 22,536 rows, only 47 nulls.

**Because there is no pollution target, the project cannot be built on the raw file alone.** The agreed fix is to fetch matching Kathmandu **PM2.5** data from the **open-meteo air-quality archive** (same data provider as the weather, so timestamps and location align perfectly) and **merge it onto the weather table by timestamp**. This is done by the provided script `build_dataset.py`.

> NOTE FOR WHOEVER RUNS THIS: `build_dataset.py` needs internet access to `air-quality-api.open-meteo.com`. It was **not** run in the environment where this brief was written because that environment blocked the domain. **Run it first, locally, before doing anything else.** It produces `kathmandu_merged.csv`.

### Caveat to disclose in the report
The PM2.5 from open-meteo is **model/reanalysis-derived, not a ground-sensor measurement.** This is acceptable for a student project but must be stated honestly. (If ground-truth is ever required, OpenAQ has real Kathmandu sensor data but needs a free API key.)

## 4. The dataset after merging (`kathmandu_merged.csv`)
Columns produced by `build_dataset.py`:
- `time` — hourly timestamp
- `temp`, `humidity`, `app_temp`, `wind10`, `wind100`, `soil_0_7`, `soil_7_28` — weather (renamed, units stripped)
- `pm2_5`, `pm10` — pollution (from open-meteo)
- `pm25_level` — categorical: Good / Moderate / Unhealthy_Sensitive / Unhealthy_or_Worse (US EPA PM2.5 breakpoints)
- `high_pollution` — binary: "High" if pm2_5 > 35.4 else "NotHigh"  ← **this is the main ARM target**

## 5. Agreed pipeline (build in this order)

### Step 0 — Sanity check
Load `kathmandu_merged.csv`. Confirm shape is close to ~22,000 rows (if it collapsed to a few hundred, the timestamp merge failed — fix parsing). Confirm `pm2_5` has real values and `high_pollution` has both classes.

### Step 1 — Cleaning
- Drop remaining nulls.
- **Consider dropping `soil_0_7` and `soil_7_28`** from the mining features. Soil moisture in an *air-pollution* project looks out of place and an examiner will ask why soil predicts air quality. Keep only if a justification is written; otherwise drop.
- Core weather features to use: `temp`, `humidity`, `wind10` (wind100 is redundant with wind10 — pick one, default wind10).

### Step 2 — Discretization (this is the real work for Apriori)
Apriori needs categorical items. Convert numeric columns into labeled bins:
- **Pollution:** already handled via `high_pollution` (and/or `pm25_level`).
- **Weather:** use **equal-frequency binning** (`pd.qcut`, tertiles) into `Low / Medium / High` for temp, humidity, wind.
  - Justification to write down: quantile binning is defensible (equal-frequency, not arbitrary cutoffs). Be ready to defend "why 3 bins" — answer: keeps rules interpretable and item count low.
- Optionally add a `season` or `month` item, BUT see Step 3 trap.
- Transform each row into a transaction, e.g. `{temp=Low, humidity=High, wind=Low, high_pollution=High}`. One-hot encode for the Apriori library (`mlxtend`).

### Step 3 — Association Rule Mining (Apriori via mlxtend)
- Run `apriori()` then `association_rules()`.
- Keep rules where **consequent = high_pollution=High** (that's what we care about).
- Filter: **lift > 1.2, confidence > 0.6, reasonable support** (e.g. > 0.02). Tune thresholds so you get a handful of meaningful rules, not thousands.
- **TRAP to avoid:** rules that just restate obvious seasonality (e.g. `{month=winter} → {high_pollution=High}`) are trivial and make the project look shallow. Prefer weather→pollution rules. If season is included, discuss it but don't let it be the whole finding.
- Report each kept rule with support / confidence / lift and a plain-English interpretation.

### Step 4 — Clustering (K-Means via scikit-learn)
- Features: the numeric weather columns (`temp`, `humidity`, `wind10`, maybe `app_temp`).
- **MANDATORY: `StandardScaler` before K-Means.** Without scaling, large-range features dominate and clusters are wrong. This is non-negotiable and will be asked about.
- Choose `k` via the **elbow method** (plot inertia vs k). Pick the elbow (likely 3–4).
- After clustering, **profile each cluster**: compute mean `pm2_5` and % `high_pollution=High` per cluster. Identify which weather cluster is the "dirty air" cluster and describe its weather signature (e.g. low wind + high humidity + cool temp).

### Step 5 — Interpretation & write-up
- Combine ARM + clustering findings: "Both methods point to [e.g. calm, humid, cool] conditions being associated with high PM2.5."
- Explicitly state limitations: modeled PM2.5, association not causation, arbitrary-ish binning, single city.

## 6. Deliverables
- Clean, commented Python (Jupyter notebook or scripts). Keep it simple and readable — this is a student project, not production.
- Plots: elbow curve, cluster profiles (mean PM2.5 per cluster bar chart), a table of top association rules.
- Short report tying findings together with the limitations section.

## 7. Suggested libraries (all standard, all "allowed")
`pandas`, `numpy`, `matplotlib`, `scikit-learn` (KMeans, StandardScaler), `mlxtend` (apriori, association_rules).

## 8. Files in this folder
- `ktmaqi.csv` — raw weather data from Kaggle (needs `skiprows=3`).
- `build_dataset.py` — fetches PM2.5 and produces `kathmandu_merged.csv`. **Run this first.**
- `PROJECT_BRIEF.md` — this file.

## 9. First actions for Claude Code
1. Run `build_dataset.py` (needs internet). Confirm `kathmandu_merged.csv` looks right (Step 0).
2. If the merge collapsed row count, fix timestamp parsing and re-run.
3. Then build Step 1 → Step 4 in a single clean notebook, simple and well-commented.
4. Keep asking: "is this the simplest thing that satisfies the requirement?" Do not over-engineer.
