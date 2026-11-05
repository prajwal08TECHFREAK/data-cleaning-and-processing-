# Cleaning & Preparing Raw Data for Machine Learning
### A hands-on lab on Indian road-accident data (2005–2023)

The five core data-prep skills, taught *and* executed on a deliberately messy dataset:

1. **Import & explore** — `shape`, `info()`, dtypes, unique counts, duplicates, `describe()`
2. **Handle missing values** — mean / median / mode / group-wise / KNN imputation
3. **Encode categoricals** — ordinal, one-hot (`drop_first`), label encoding
4. **Scale numerics** — Standard / MinMax / Robust scalers, and `log1p` for skew
5. **Find & remove outliers** — boxplots, IQR fences, and the delete-vs-clip decision

**Live pages** (once GitHub Pages is enabled): `docs/` — walkthrough, results report, notebook export.

---

## Quick start

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### See the results without running anything
Open **`docs/run_report.html`** — the full run: every number, table and figure.

### Reproduce the whole thing

```bash
python3 scripts/01_make_dataset.py   # rebuild the messy CSV (verifies MoRTH totals)
python3 scripts/run_pipeline.py      # the five steps -> data/ + output/
python3 scripts/build_report.py      # -> docs/run_report.html
```

### Work through the lesson interactively

```bash
jupyter notebook clean_and_prepare.ipynb
```

---

## What is in the repo

```
.
├── clean_and_prepare.ipynb          # the lesson: 66 cells, executed, all plots rendered
├── data/
│   ├── india_road_accidents_raw.csv       # the messy input (10,465 x 14)
│   ├── india_road_accidents_clean.csv     # output of the notebook (9,666 x 19)
│   └── road_accidents_clean_final.csv     # output of the pipeline (9,666 x 34, model-ready)
├── scripts/
│   ├── 01_make_dataset.py           # generates + deliberately corrupts the data
│   ├── 02_build_notebook.py         # builds the notebook programmatically
│   ├── run_pipeline.py              # the five steps, executed in order
│   └── build_report.py              # summary.json + figures -> HTML report
├── output/
│   ├── run_log.txt                  # full console log
│   ├── summary.json                 # every statistic, machine-readable
│   └── figs/                        # 8 figures (PNG)
├── docs/                            # GitHub Pages
│   ├── index.html                   # visual walkthrough of the five steps
│   ├── run_report.html              # results report from the actual run
│   └── notebook.html                # read-only notebook export
├── requirements.txt
└── .gitignore
```

---

## The five steps, as executed

| Step | Key result on this data |
|---|---|
| 1. Explore | 5 of 6 numeric columns arrive as `object`; 60 spelling variants of 30 states; 205 duplicate rows |
| 2. Missing | 7 columns at 4–11% missing → median for skewed counts, group-wise median for driver age, mode for categoricals → **0 remaining** |
| 3. Encoding | vehicle_type 8 → 6, state 60 → 30, 16 one-hot dummies, ordinal severity, label-encoded state |
| 4. Scaling | skew **4.56 → 0.01** with `log1p`; row distance **16,604 → 11.18** after standardising; 7 scaled columns |
| 5. Outliers | 594 rows deleted (impossible values only), extreme-but-real tails clipped; 10,260 → **9,666 rows** |

### Three findings worth more than the code

* **The IQR rule flagged 10% of `accidents` rows** — those are Tamil Nadu, Madhya Pradesh and Uttar
  Pradesh, the largest states in the study. Deleting them would gut the dataset. So: delete what is
  *impossible* (speed limit 999, age −1), clip what is merely *extreme*.
* **Imputation shrinks variance** (4912 raw → ~4445 after any method), and mean/median both leave a
  visible spike at one value. KNN-5 spreads its guesses and looks smoother.
* **Scalers do not fix skew.** Standard/MinMax/Robust leave the shape identical and only relabel the
  axis. Only the log transform changes the shape.

### The one rule that governs everything

> Learn every cleaning statistic — medians, modes, scaler means, IQR fences — from the **training set
> only**. Fit on train, transform test.

`scripts/run_pipeline.py` computes them on the whole dataset, which is correct for a *cleaning report*
and **wrong for a model**. The final section of `clean_and_prepare.ipynb` shows the leak-free version
as an `sklearn` `Pipeline` (`SimpleImputer → StandardScaler`, `SimpleImputer → OneHotEncoder`) fitted
on `X_train` only.

---

## Publish the pages

Settings → Pages → Source: **Deploy from a branch** → Branch: `main`, folder: **`/docs`** → Save.

Then `https://<your-username>.github.io/<repo-name>/` serves the walkthrough, with the results report
at `/run_report.html` and the notebook at `/notebook.html`.

---

## About the data

Yearly national totals are the published **MoRTH / NCRB "Road Accidents in India"** figures —
4,80,583 accidents, 1,72,890 deaths and 4,62,825 injuries in 2023; 2020 reflects the COVID lockdown
drop. Row-level detail (state × year × road type × vehicle type, weather, cause, severity) is a
realistic reconstruction created for teaching and **deliberately corrupted** so there is something to
clean. Yearly sums still reconcile exactly to the official series, so aggregates you compute are
defensible.

For real analysis use the official releases at `morth.nic.in` and `data.gov.in`, or NCRB's
*Accidental Deaths & Suicides in India*.
