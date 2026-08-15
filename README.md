# Discovery of Meteorological Patterns Associated with High PM2.5 Pollution Episodes in Kathmandu

This reproducible university data-mining project asks: **Which combinations of meteorological conditions are associated with high PM2.5 pollution episodes in Kathmandu?**

The notebook separates two complementary methods:

- **K-Means clustering** discovers recurring multivariate meteorological regimes and then profiles the observed PM2.5 in each regime. PM2.5 is deliberately excluded from the clustering inputs so the weather regimes are not defined by the outcome being described.
- **Association-rule mining** finds interpretable combinations of categorized weather conditions associated with Unhealthy-or-worse PM2.5 concentration categories.

## Project structure

```text
.
|-- cleaned_dataset.csv
|-- notebooks/
|   `-- analysis.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- clustering.py
|   `-- association_rules.py
|-- requirements.txt
`-- README.md
```

## Run the analysis

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

Run all notebook cells from top to bottom. Figures and conclusions are generated from `cleaned_dataset.csv`; no empirical conclusion is hard-coded.

## Methodological notes

- Duplicate rows are removed. Short, internal numeric gaps are time-interpolated; remaining gaps use month-hour medians, then a global median fallback.
- Kathmandu seasons are defined as Winter (Dec-Feb), Pre-monsoon (Mar-May), Monsoon (Jun-Sep), and Post-monsoon (Oct-Nov).
- Weather variables use dataset tertiles for Low/Medium/High bins. This makes labels relative to the observed Kathmandu dataset, and the notebook prints the exact cut points.
- PM2.5 labels use US EPA 24-hour concentration breakpoints. Because the records are hourly, these are **screening categories**, not official daily AQI classifications. "High" in the rule-mining analysis means Unhealthy, Very Unhealthy, or Hazardous (PM2.5 >= 55.5 micrograms/m3).
- Association does not establish causation. Rules describe co-occurrence in this dataset and should be validated on independent data.

