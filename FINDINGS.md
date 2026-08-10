# Exploration Findings — Kathmandu AQI Data Mining

Pre-analysis notes from inspecting the raw data and the open-meteo air-quality API
**before** any of the pipeline in `PROJECT_BRIEF.md` was built. Nothing here required
pandas — all numbers below came from stdlib `csv` + `urllib` on the raw files, so they
are independent of the pipeline and can be used to sanity-check it later.

Date of exploration: 2026-08-10.

---

## 1. Raw weather file (`ktmaqi.csv`) is cleaner than the brief suggests

| Property | Value |
|---|---|
| Data rows (after `skiprows=3`) | 22,536 |
| Time span | 2022-08-05T00:00 → 2025-02-28T23:00 (GMT) |
| Expected hours in that span | 22,536 |
| Missing hours / gaps | **0** |
| Duplicate timestamps | **0** |
| Rows with any NaN | 47 |
| Complete rows | **22,489** |

The series is perfectly continuous hourly — no gaps, no duplicates. The 47 nulls are
**not scattered**; they are one contiguous block at the tail
(`2025-02-27T01:00` → `2025-02-28T23:00`). So `dropna()` simply trims the final two
days rather than punching holes in the middle.

Column summary (complete rows only):

| col | min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|---|
| temp (°C) | 1.30 | 13.30 | 19.00 | 22.10 | 32.60 | 17.92 |
| humidity (%) | 7.00 | 61.00 | 79.00 | 92.00 | 100.00 | 74.46 |
| app_temp (°C) | -1.60 | 12.50 | 19.20 | 25.10 | 35.80 | 18.67 |
| wind10 (km/h) | 0.00 | 2.40 | 3.60 | 5.20 | 18.80 | 4.02 |
| wind100 (km/h) | 0.00 | 2.60 | 4.30 | 6.60 | 26.80 | 5.01 |
| soil_0_7 | 0.10 | 0.21 | 0.30 | 0.40 | 0.43 | 0.29 |
| soil_7_28 | 0.15 | 0.22 | 0.30 | 0.40 | 0.43 | 0.31 |

---

## 2. The timestamp merge will not collapse

The brief's Step 0 warns the merge might collapse to a few hundred rows. Checked
directly against the API rather than assumed — it will not:

- Requesting the full range returns **22,536 hourly values, zero nulls**, spanning
  exactly `2022-08-05T00:00` → `2025-02-28T23:00`.
- Time strings are **byte-identical in format** to the weather file
  (`2022-08-05T00:00`), both GMT — no timezone or parsing mismatch.
- An in-memory join yields **22,489 rows = a 100% match** against the complete
  weather rows.

Also: `air-quality-api.open-meteo.com` **is reachable** from this machine (HTTP 200).
The brief's note about the domain being blocked is stale and does not apply here.

---

## 3. Target variable is well balanced

PM2.5 distribution: min 0.2, p25 18.2, **median 27.7**, p75 42.8, p95 82.4, max 170.4 µg/m³.

| Class | count | share |
|---|---|---|
| Good (≤12) | 2,088 | 9.3% |
| Moderate (12–35.4) | 12,515 | 55.5% |
| Unhealthy_Sensitive (35.4–55.4) | 4,722 | 21.0% |
| Unhealthy_or_Worse (>55.4) | 3,211 | 14.2% |

`high_pollution` (PM2.5 > 35.4) splits **35.2% High / 64.8% NotHigh**. No rare-class
problem — ARM support thresholds can be set normally.

---

## 4. ⚠ The main analytical risk (differs from the one the brief flags)

`PROJECT_BRIEF.md` §Step 3 warns against trivial `{month=winter} → {high_pollution=High}`
rules. The real issue is that **temperature is itself the seasonality proxy**, and it
overwhelms the other two features.

Baseline P(high_pollution=High) = **0.351**.

Single-item tertile (equal-frequency) results:

| feature | Pearson r vs pm2_5 | tertile | n | confidence | lift |
|---|---|---|---|---|---|
| **temp** | **−0.607** | Low (≤15.5) | 7,557 | 0.678 | **1.93** |
| | | Medium | 7,457 | 0.251 | 0.72 |
| | | High (>21.1) | 7,475 | 0.120 | 0.34 |
| **humidity** | +0.074 | Low (≤68) | 7,554 | 0.326 | 0.93 |
| | | Medium | 7,731 | 0.412 | 1.17 |
| | | High (>88) | 7,204 | 0.312 | 0.89 |
| **wind10** | −0.031 | Low (≤2.9) | 8,255 | 0.346 | 0.98 |
| | | Medium | 6,741 | 0.368 | 1.05 |
| | | High (>4.6) | 7,493 | 0.342 | 0.97 |

**Wind10 is inert on its own** — all three tertiles sit at lift 0.97–1.05. Humidity is
weak and non-monotonic (Medium beats both Low and High).

Consequence: a plain Apriori run on `{temp, humidity, wind}` will return one dominant
rule, `{temp=Low} → {high_pollution=High}`, and the humidity/wind items will mostly ride
along as passengers. The Step 5 conclusion sketched in the brief ("calm, humid, cool
conditions") is **not** what the single-item data supports.

---

## 5. The way out — a genuine 2-item interaction

Still inside the two allowed algorithms; no new method needed. Conditioning on cold
hours, wind separates properly:

Within `temp=Low` (n = 7,557), baseline confidence = **0.678**

| itemset | n | confidence | lift vs global | lift vs temp=Low |
|---|---|---|---|---|
| temp=Low & wind10=Low | 2,566 | **0.757** | **2.16** | **1.12** |
| temp=Low & wind10=Medium | 2,699 | 0.650 | 1.85 | 0.96 |
| temp=Low & wind10=High | 2,292 | 0.622 | 1.77 | 0.92 |
| temp=Low & humidity=Low | 1,825 | 0.561 | 1.60 | 0.83 |
| temp=Low & humidity=Medium | 3,385 | 0.725 | 2.07 | 1.07 |
| temp=Low & humidity=High | 2,347 | 0.701 | 2.00 | 1.03 |

**Reading:** stagnant air matters, but only once it is already cold — 0.757 vs 0.622
confidence between calm and windy cold hours. That is a defensible finding rather than
a restatement of "winter is polluted".

Two practical notes for Step 3:
- Do **not** set thresholds so tight that only the `{temp=Low}` rule survives. The
  interesting 2-itemsets have support ≈ 2,566/22,489 ≈ **0.114**, comfortably above the
  brief's 0.02 floor, so this is about not over-filtering on lift/confidence.
- Report **lift against the `temp=Low` baseline alongside global lift**. Global lift
  alone makes every cold itemset look strong; the conditional column is what shows wind
  is actually contributing.

Humidity's conditional signal is weak (1.03–1.07) — worth reporting honestly as a
near-null result rather than overselling it.

---

## 6. Bonus signal not mentioned in the brief — diurnal cycle

| hour band (GMT) | n | confidence | lift |
|---|---|---|---|
| 00–05 | 5,623 | 0.334 | 0.95 |
| 06–11 | 5,622 | 0.195 | **0.56** |
| 12–17 | 5,622 | 0.457 | **1.30** |
| 18–23 | 5,622 | 0.418 | 1.19 |

Strong and clean. Adding an hour-band item is optional — it carries the same
triviality caveat as `season`/`month` per the brief's Step 3 trap, so if included it
should be discussed but not presented as the headline finding.

Note the file is GMT (`utc_offset_seconds = 0`), while Nepal is UTC+05:45. If hour-band
items are used, convert to local time first or the bands will be mislabelled in the
writeup.

---

## 7. Environment status

- Python 3.12.3, pip 24.0, `venv` available.
- **pandas, numpy, matplotlib, scikit-learn, mlxtend are all missing**; only `requests`
  is installed. Ubuntu 24.04 pip is externally-managed → use a `.venv`
  (see `requirements.txt`).
- `kathmandu_merged.csv` does **not** exist yet; `build_dataset.py` has not been run.
- The initial commit message mentions an "ARM/clustering notebook" but no notebook was
  ever committed — only `PROJECT_BRIEF.md` and `build_dataset.py`.

---

## 8. Caveats carried forward

- PM2.5 is **model/reanalysis-derived** (open-meteo/CAMS), not a ground sensor. Must be
  stated in the report, per brief §3.
- Association, not causation — wording throughout should stay "associated with" /
  "co-occurs with".
- Tertile cut points are data-driven (equal-frequency) but the choice of *three* bins is
  still a judgement call; defend as "keeps rules interpretable and item count low".
- Single city, ~2.5 years, single location grid cell.
