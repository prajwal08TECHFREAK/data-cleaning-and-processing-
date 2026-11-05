"""
02_build_notebook.py - generates clean_and_prepare.ipynb (executed, with outputs)
Run: python3 scripts/02_build_notebook.py
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ---------------------------------------------------------------------------
md("""
# Cleaning & Preparing Raw Data for Machine Learning
### A hands-on lab on Indian road-accident data (2005 – 2023)

**What you will be able to do by the end of this notebook**

| # | Skill | Why it matters for ML |
|---|-------|-----------------------|
| 1 | Import a CSV and profile it (`shape`, `info`, `dtypes`, `describe`) | You cannot fix what you cannot see. Garbage dtypes silently break every model. |
| 2 | Handle missing values (drop / mean / median / mode / group-wise / KNN imputation) | Most algorithms refuse `NaN` or quietly learn the wrong thing from it. |
| 3 | Encode categorical features (one-hot, ordinal, label, target) | Models only multiply numbers — text must become numbers *without* inventing an order that isn't there. |
| 4 | Normalise / standardise numerical features | A feature measured in lakhs must not bully a feature measured in tens. |
| 5 | Visualise outliers with boxplots and remove them | One bad row can drag a regression line across the whole state. |

**The dataset** — `data/india_road_accidents_raw.csv`, one row per *(state × year × road type × vehicle type)* accident cluster.
The yearly national totals are the real MoRTH / NCRB published figures (e.g. **4,80,583 accidents and 1,72,890 deaths in 2023**).
The row-level detail is a realistic reconstruction, and the file has been **deliberately corrupted** the way real downloads are:
messy category spellings, `NA` / `N/A` / `-` / `null` / `Unknown`, numbers saved as text (`"1,234"`), sentinel codes like `999`,
duplicated rows, and a handful of genuinely impossible values. Your job is to clean it.

**The golden rule of this whole notebook**

> Learn every cleaning statistic (mean, median, scaler, encoder, outlier bound) from the **training set only** —
> fit on train, `transform` on test. Otherwise information leaks from test into train and your offline score becomes a lie.

Let's set up the environment first.
""")

code("""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

SEVERITY_ORDER = {"Minor": 0, "Serious": 1, "Fatal": 2}

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (11, 4.5)
RANDOM_STATE = 42
print("pandas", pd.__version__, "| numpy", np.__version__)
""")

# ---------------------------------------------------------------- STEP 1
md("""
---
## Step 1 — Import the dataset and explore basic info

Before touching a single value, answer four questions:

1. **How big is it?** → `shape`
2. **What type is each column?** → `dtypes` / `info()`. A number showing up as `object` is a red flag.
3. **How dirty is it?** → `isna().sum()`, `nunique()`, `value_counts()`
4. **Do the numbers make physical sense?** → `describe()`

`dtype` is the single most informative thing about a column. `accidents` being `object` instead of `int64` means
there is text hiding inside it, and *every* calculation you run on it will fail or silently mislead.
""")

code("""
df = pd.read_csv("data/india_road_accidents_raw.csv")

print("Rows, columns:", df.shape)
df.head()
""")

code("""
# info() = dtype + non-null count per column, in one shot
df.info()
""")

code("""
# Which columns are numeric vs text? And how many distinct values does each hold?
profile = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "missing_%": (df.isna().mean() * 100).round(1),
    "n_unique": df.nunique(),
    "example": [df[c].dropna().iloc[0] for c in df.columns],
})
profile
""")

md("""
Look at that table: `accidents`, `deaths`, `injuries`, `vehicles_involved`, `speed_limit_kmph` and `driver_age_median`
are all `object`. They are numbers trapped in text. Let's prove it:
""")

code("""
# The tell-tale sign: numbers stored as text sort alphabetically
print("dtype:", df[\"deaths\"].dtype)
print(df[\"deaths\"].astype(str).str.sort_values if False else "")
print(sorted(df[\"deaths\"].astype(str).unique(), reverse=True)[:6])

# ...and string methods work on them, which is a dead giveaway
print("\\ncontains commas:", df[\"deaths\"].astype(str).str.contains(\",\").sum(), "rows")
""")

code("""
# The text columns hide junk spellings of the same category
for col in ["state", "road_type", "vehicle_type", "weather", "alcohol_involved"]:
    vals = sorted(df[col].dropna().astype(str).unique())
    print(f"{col:18s} ({len(vals):2d} variants): {vals}")
""")

md("""
`vehicle_type` shows `Two-wheeler` **and** `two wheeler`, `road_type` has trailing spaces,
`state` has `UTTAR PRADESH` next to `Uttar Pradesh`, and `weather` mixes `Clear` with `clear`.
If you one-hot encode now you get **two separate columns for the same thing** and each gets half the data.

### Two more cheap diagnostics worth running every single time
""")

code("""
# 1. Duplicates: the same row downloaded twice
print("fully duplicated rows:", df.duplicated().sum())

# 2. describe() on numeric columns - but first we must coerce
num_check = df[[\"accidents\", \"deaths\", \"injuries\", \"speed_limit_kmph\", \"driver_age_median\"]].apply(
    pd.to_numeric, errors=\"coerce\")
num_check.describe().round(2)
""")

md("""
`describe()` is the fastest way to catch impossible data. In this raw file the tells are:

* `speed_limit_kmph` **max 999** while its 75th percentile is only 100 — a sentinel code, not a speed
* `driver_age_median` **min -1 and max 999** — negative age, impossible age
* `accidents` max in the tens of thousands — plausible for a big state, but we should check
* Coercing to numeric also produced new `NaN`s: those are the `"1,234"` and `NA` strings.

**Step 1 checklist** ✅ read → `shape` → `info()` → dtypes → missing % → unique counts → duplicates → `describe()`.
""")

# ---------------------------------------------------------------- STEP 2
md("""
---
## Step 2 — Handle missing values

There are only **four** legitimate strategies. Choosing between them is a judgement call, not a formula:

| Strategy | Use when | Cost |
|----------|----------|------|
| **Drop rows / columns** | Missingness is tiny (`<5%`) and random, or the column is mostly empty (`>50%`) | Loses data; biases the sample if missingness is *not* random |
| **Mean** | Numeric, roughly symmetric, no outliers | Shrinks variance; wrecked by outliers |
| **Median** | Numeric, skewed or with outliers (our accident counts!) | Still shrinks variance |
| **Mode** | Categorical | Can over-represent the majority class |
| **Group-wise (conditional) mean/median** | A related column predicts the missing one (e.g. fill speed by road type) | Much more accurate — this is what practitioners actually do |
| **KNN / iterative (MICE)** | You have many correlated features and time to spare | Leakage risk, slow, still just an educated guess |

**The first job is not to fill anything — it is to normalise the *representation* of "missing".**
This file uses `NA`, `N/A`, ``, `Unknown`, `-` and `null` for the same concept. If you impute now, `"-"` is treated as a
valid category and you get a category called `"-"` in your model.
""")

code("""
missing_tokens = [\"NA\", \"N/A\", \"\", \"Unknown\", \"null\", \"-\", \"nan\", \"None\"]
df_clean = df.replace(missing_tokens, np.nan)
df_clean = df_clean.apply(lambda s: s.str.strip() if s.dtype == \"object\" else s)

# how much is missing, for real this time
miss = df_clean.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
pd.DataFrame({\"missing\": miss, \"percent\": (miss / len(df_clean) * 100).round(2)})
""")

code("""
# Visualise missingness before deciding: a bar chart is enough for this dataset.
# (For wide datasets use sns.heatmap(df.isna().T, cbar=False) instead.)
ax = (miss / len(df_clean) * 100).plot.barh(color=\"#c44e52\")
ax.set_xlabel(\"% missing\")
ax.set_title(\"Missing values per column, after normalising the NA tokens\")
ax.invert_yaxis()
plt.tight_layout(); plt.show()
""")

md("""
Every column sits under 11%, so **dropping rows would be wasteful** — 11% here, 5% there, and a row missing two
fields is gone entirely. We will impute, but column by column with the right statistic.

### 2a. Fix the dtypes first (you cannot take a mean of text)

`"1,234"` must become `1234`, and `"NA"` must become `NaN`. `pd.to_numeric(..., errors="coerce")` does both in one line:
""")

code("""
# Try the arithmetic first, on the raw frame, and watch it fail:
try:
    df[\"accidents\"].mean()
except TypeError:
    print(\"TypeError: pandas cannot compute a mean() of a text column.\")
print(\"df['accidents'].dtype is\", df[\"accidents\"].dtype, \"-> every statistic you compute is wrong.\")
print(\"Worse, the failure is silent in some operations: .sum() concatenates the strings.\")
""")

code("""
NUM_COLS = [\"accidents\", \"deaths\", \"injuries\", \"vehicles_involved\", \"speed_limit_kmph\", \"driver_age_median\"]

for col in NUM_COLS:
    df_clean[col] = pd.to_numeric(
        df_clean[col].astype(str).str.replace(\",\", \"\", regex=False).str.strip(),
        errors=\"coerce",
    )

df_clean.dtypes
""")

md("""
### 2b. Remove rows that can never be repaired

Some missingness is *structural*. A row with no `accidents` count, or a duplicate of another row,
carries no information and no imputation can invent it honestly. Drop those first — it is the cheapest
and least controversial cleaning step.
""")

code("""
before = len(df_clean)

# (i) exact duplicates from a double download
df_clean = df_clean.drop_duplicates()

# (ii) the target itself must be present
TARGET = \"severity\"
df_clean = df_clean.dropna(subset=[TARGET, \"accidents\", \"year\", \"state\"])

print(f\"dropped {before - len(df_clean)} rows ({(before-len(df_clean))/before:.1%}) -> {len(df_clean)} rows remain\")
""")

md("""
### 2c. Compare mean vs median before you pick one

Rule of thumb: **if mean and median are far apart, the distribution is skewed → use the median.**
Accident counts are heavily right-skewed (a few huge states, many small ones), so the median is the honest choice.
""")

code("""
stats = pd.DataFrame({
    \"mean\": df_clean[NUM_COLS].mean(),
    \"median\": df_clean[NUM_COLS].median(),
    \"skew\": df_clean[NUM_COLS].skew(),
})
stats[\"mean_vs_median_%_gap\"] = ((stats[\"mean\"] - stats[\"median\"]).abs() / stats[\"median\"] * 100).round(1)
stats.round(2)
""")

md("""
`accidents`, `deaths` and `injuries` have a mean 2–3× their median: textbook right skew.
`driver_age_median` is nearly symmetric (mean ≈ median), so the mean would be fine there — but the median harms nothing,
so we keep it simple and use the median for all count-like columns.

### 2d. Three imputation methods, compared on the same column

Let's fill `driver_age_median` three ways and look at what each does to the distribution, then do the same for a categorical.
""")

code("""
col = \"driver_age_median\"
work = df_clean[[col]].copy()

mean_imp = work[col].fillna(work[col].mean())
median_imp = work[col].fillna(work[col].median())
knn_imp = pd.Series(KNNImputer(n_neighbors=5).fit_transform(work)[:, 0], index=work.index)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharex=True)
for ax, (name, s) in zip(axes, [(\"mean\", mean_imp), (\"median\", median_imp), (\"KNN-5\", knn_imp)]):
    ax.hist(s, bins=35, color=\"#4c72b0\")
    ax.set_title(f\"{name} imputation (median={s.median():.1f})\")
    ax.set_xlabel(col)
plt.suptitle(\"Imputation method changes the shape of the data, not just the NaN count\", y=1.03)
plt.tight_layout(); plt.show()

print(f\"filled with mean   -> variance {mean_imp.var():.2f}\")
print(f\"filled with median -> variance {median_imp.var():.2f}\")
print(f\"filled with KNN    -> variance {knn_imp.var():.2f}\")
print(f\"original variance  -> {work[col].var():.2f}  (all methods shrink it slightly)\")
""")

md("""
Read those three histograms closely. Mean and median fill produce a **spike at one single value** — you have just
invented hundreds of identical drivers, which flattens the real spread. KNN spreads its guesses across the plausible
range, so it looks smoother. Here the variances barely differ (4444 vs 4447 vs 4444, against an original 4911) because
this column is nearly symmetric and only ~9.5% missing; on a **skewed** column with heavy missingness, mean-imputation
visibly drags the whole distribution toward the middle.

**Group-wise imputation is strictly better when a correlated column exists.**
A truck's driver age is not drawn from the same distribution as a two-wheeler rider's, so fill each vehicle type
with *its own* median. Same amount of code, much less distortion:
""")

code("""
# Before: one global median. After: the median of that vehicle's own group.
df_clean[\"driver_age_median\"] = df_clean.groupby(\"vehicle_type\")[\"driver_age_median\"].transform(
    lambda s: s.fillna(s.median())
)
# any stragglers (a group that was entirely NaN) get the global median
df_clean[\"driver_age_median\"] = df_clean[\"driver_age_median\"].fillna(df_clean[\"driver_age_median\"].median())

table = df_clean.groupby(\"vehicle_type\")[\"driver_age_median\"].agg([\"median\", \"count\"]).round(1)
print(table.to_string())
print(\"\u26a0 Count the rows: 8 groups, but India has 6 vehicle types. 'Two-wheeler'/'two wheeler' and\")
print(\"  'Car / Taxi / Van'/'Car/Taxi/Van' are the SAME vehicle spelled differently, so they were\")
print(\"  imputed separately. Lesson: tidy your categories BEFORE you impute, not after.\")
""")

code("""
# Numeric count columns: median is the safe default for skewed data
for col in [\"accidents\", \"deaths\", \"injuries\", \"vehicles_involved\", \"speed_limit_kmph\"]:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Categorical columns: mode (most frequent) - or an explicit \"Unknown\" bucket
for col in [\"weather\", \"accident_cause\", \"alcohol_involved\"]:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

print(\"remaining missing values:\", int(df_clean.isna().sum().sum()))

# Note: \"Unknown\" as its own category is often BETTER than the mode, because
# missingness itself can be informative. It costs one extra dummy column.
""")

md("""
### ⚠️ The mistake almost everyone makes

The code above imputed using the median of the **whole dataset** — including the test set. In a real project that is
**data leakage**: at prediction time you will not have the test set's median. Do it inside a `Pipeline` instead
(shown at the end of this notebook) so the imputer is fitted on `X_train` only.

**Step 2 checklist** ✅ normalise NA tokens → fix dtypes → drop unrepairable rows/duplicates →
compare mean vs median → group-wise fill → verify zero `NaN` remain.
""")

# ---------------------------------------------------------------- STEP 3
md("""
---
## Step 3 — Convert categorical features into numbers

| Encoding | Produces | Use when |
|----------|----------|----------|
| **One-hot (dummy)** | 0/1 column per category | **Nominal** data (no order): state, road_type, weather. The default choice. |
| **Ordinal** | One integer column, 0..k-1 | **Ordered** data: severity Minor < Serious < Fatal |
| **Label / factorize** | One integer column, arbitrary order | Tree models, or when cardinality is huge. **Never for linear models** — it invents a fake ranking. |
| **Target / mean encoding** | Category → mean of target | Very high cardinality (e.g. 700+ states/districts). Powerful but leaks badly if done outside CV. |

The trap to avoid: one-hot encoding `state` (30 values) and `vehicle_type` (6) and `weather` (4) all at once, then
feeding thousands of columns into a linear model. Watch the dimension count as we go.
""")

code("""
# The fix that actually works is not a regex - it is a LOOKUP TABLE.
# Normalise case/whitespace first, then map every known variant onto one canonical label.
def tidy_category(series):
    return series.astype(str).str.strip().str.lower().str.replace("\\s+", " ", regex=True)

CANONICAL = {
    "vehicle_type": {"two wheeler": "Two-Wheeler", "two-wheeler": "Two-Wheeler",
                     "car/taxi/van": "Car / Taxi / Van", "car / taxi / van": "Car / Taxi / Van",
                     "auto-rickshaw": "Auto-Rickshaw", "truck / lorry": "Truck / Lorry",
                     "bus": "Bus", "pedestrian / cyclist": "Pedestrian / Cyclist"},
    "road_type":    {"national highway": "National Highway", "state highway": "State Highway",
                     "other roads": "Other Roads"},
    "state":        {"jammu & kashmir": "Jammu And Kashmir", "other uts": "Other UTs"},
}

for col in ["state", "road_type", "vehicle_type", "weather", "accident_cause", "alcohol_involved"]:
    df_clean[col] = tidy_category(df_clean[col])
    df_clean[col] = df_clean[col].replace(CANONICAL.get(col, {})).str.title()

for col in [\"state\", \"road_type\", \"vehicle_type\", \"weather\"]:
    print(f\"{col:15s} -> {df_clean[col].nunique():2d} clean categories\")
print(df_clean[\"vehicle_type\"].value_counts().to_string())
""")

md("""
### 3a. Ordinal encoding — for features that genuinely have an order

`severity` is the natural ML target here (predict how bad an accident will be from its circumstances).
It has a real order, so an explicit mapping is clearer and safer than `LabelEncoder` (whose alphabetical
answer, `Fatal=1, Minor=2, Serious=0`, would be nonsense).
""")

code("""
# SEVERITY_ORDER lives next to the imports so the same mapping is reused in the pipeline.
df_clean[\"severity_code\"] = df_clean[\"severity\"].map(SEVERITY_ORDER)

print(df_clean[[\"severity\", \"severity_code\"]].drop_duplicates().sort_values(\"severity_code\").to_string(index=False))
print(df_clean[\"severity_code\"].value_counts(normalize=True).round(3).to_string())
""")

md("""
Note the imbalance: `Serious` is the largest class at ~48%, `Fatal` the smallest at ~21%. Predicting `Serious` for
every single row therefore scores **0.48 accuracy** while learning nothing at all. Hold on to that number for the model
section — **for imbalanced data, accuracy alone is a misleading metric**; look at per-class precision and recall too.

### 3b. One-hot encoding — for features with no order

`road_type`, `weather`, `vehicle_type` are nominal. We use `pd.get_dummies` here for readability;
in a pipeline you would use `OneHotEncoder(handle_unknown=\"ignore\")` so that an unseen category at prediction time
becomes all-zeros instead of an error.
""")

code("""
NOMINAL = [\"road_type\", \"weather\", \"vehicle_type\", \"accident_cause\", \"alcohol_involved\"]

print(\"columns before one-hot:\", df_clean.shape[1])
df_encoded = pd.get_dummies(df_clean, columns=NOMINAL, drop_first=True, dtype=int)
# .duplicated(axis=1) is a cheap safety net: it removes the case where two dummies
# encode the same information (a classic symptom of dirty category spellings).
df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]
print(\"columns after  one-hot:\", df_encoded.shape[1])
print(\"\\nnew dummy columns:\")
print([c for c in df_encoded.columns if any(c.startswith(n + \"_\") for n in NOMINAL)])
""")

md("""
`drop_first=True` drops one dummy per feature. Why: if `road_type` has 3 categories and you keep all 3 dummies,
then `NH + SH + Other = 1` always — perfectly collinear features. Linear models and any interpretation of
coefficients break. (Tree models don't care, so for them you can leave `drop_first=False`.)

### 3c. Should we one-hot `state`?

It has 30 categories. Doing so adds 29 columns that are mostly noise for a small dataset — the classic
**curse of dimensionality**. Two better options:

* **Label encode it** if the model is tree-based (LightGBM/XGBoost/CatBoost handle integer-coded categories well)
* **Target encode it** for linear models (replace the state with the state's average fatality rate — computed inside CV!)

We will label-encode it and then compare a linear model with vs without it.
""")

code("""
df_encoded[\"state_code\"] = df_encoded[\"state\"].astype(\"category\").cat.codes

# sanity check: the mapping is arbitrary but consistent
print(df_encoded[[\"state\", \"state_code\"]].drop_duplicates().sort_values(\"state_code\").head(8).to_string(index=False))
print(\"...\\nA linear model would read state_code as a real number (24 > 3), which is meaningless.\")
print(\"That is exactly why label encoding is for TREES, not for linear/neural models.\")
""")

md("""
### 3d. Same thing, the production way — `ColumnTransformer`

Hand-encoding is fine for learning. In real code you bundle all of it into a transformer that is fitted on train
and reused on test. This is what we use at the end for the model.
""")

code("""
cat_pipe_preview = ColumnTransformer(
    transformers=[
        (\"nominal\", OneHotEncoder(handle_unknown=\"ignore\", drop=\"first\"), NOMINAL),
        (\"ordinal\", OrdinalEncoder(categories=[[\"Minor\", \"Serious\", \"Fatal\"]]), [\"severity\"]),
    ],
    remainder=\"passthrough\",
)
Xt = cat_pipe_preview.fit_transform(df_clean[NOMINAL + [\"severity\"] + NUM_COLS])
print(\"input columns :\", len(NOMINAL) + 1 + len(NUM_COLS))
print(\"output columns:\", Xt.shape[1], \"(one-hot expansion happened automatically)\")
""")

md("""
**Step 3 checklist** ✅ standardise spellings first → ordinal-encode ordered features →
one-hot nominal features with `drop_first` → handle high cardinality with labels/target encoding →
never use label encoding on a linear model.
""")

# ---------------------------------------------------------------- STEP 4
md("""
---
## Step 4 — Normalise / standardise numerical features

| Scaler | Formula | Output range | Use when |
|--------|---------|--------------|----------|
| **StandardScaler** (z-score) | `(x - mean) / std` | mean 0, std 1 | **Default.** Needed by linear models, SVM, k-NN, PCA, neural nets |
| **MinMaxScaler** | `(x - min) / (max - min)` | exactly [0, 1] | You need a bounded range (image pixels, some neural nets) |
| **RobustScaler** | `(x - median) / IQR` | not bounded | **Outliers present** — this is the one for our data |
| **Log / sqrt transform** | `log(1 + x)` | compresses the tail | Heavy right skew (accident counts!) — fixes skew *as well as* scale |

Two facts people get wrong:

1. **Tree models (Random Forest, XGBoost) do not need scaling.** They split on thresholds, and thresholds are
   invariant to monotone transforms. Scaling them is harmless but pointless.
2. **Scaling must be fitted on train only.** If you `fit` the scaler on the full data, the test set's minimum
   and maximum leak into training and your reported score is optimistic.
""")

code("""
scale_cols = NUM_COLS + [\"year\"]
raw_num = df_clean[scale_cols]

comparison = pd.DataFrame({
    \"raw\": raw_num[\"accidents\"],
    \"standard\": StandardScaler().fit_transform(raw_num[[\"accidents\"]]).ravel(),
    \"minmax\": MinMaxScaler().fit_transform(raw_num[[\"accidents\"]]).ravel(),
    \"robust\": RobustScaler().fit_transform(raw_num[[\"accidents\"]]).ravel(),
    \"log1p\": np.log1p(raw_num[\"accidents\"]),
})
comparison.describe().round(3)
""")

code("""
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for ax, col in zip(axes.ravel(), comparison.columns):
    ax.hist(comparison[col], bins=40, color=\"#55a868")
    ax.set_title(f\"{col}\\nmean={comparison[col].mean():.2f} std={comparison[col].std():.2f}\")
plt.suptitle(\"accidents - one column, five representations (same shape, different axis)\", y=1.03)
plt.tight_layout(); plt.show()
""")

md("""
Look carefully: the four scalers produce **the same shape**, only relabelled axes; skew is untouched.
The log transform is the only one that changes the *shape* — and for accident counts that is usually what you want.

### Scale matters only when distances matter

A quick demonstration of why mixed units are dangerous. `accidents` ranges into the thousands;
`vehicles_involved` ranges 1–6. In raw Euclidean distance, `accidents` is ~1000× more influential, purely by accident of unit.
""")

code("""
pair = df_clean[[\"accidents\", \"vehicles_involved\"]].copy()
d_raw = np.linalg.norm(pair.iloc[0] - pair.iloc[1])

pair_s = StandardScaler().fit_transform(pair)
d_scaled = np.linalg.norm(pair_s[0] - pair_s[1])

print(f\"distance in raw units    : {d_raw:10.2f}   <- dominated by 'accidents'\")
print(f\"distance after scaling   : {d_scaled:10.2f}   <- both features now get a fair say\")

print(\"\\nfeature ranges (why the above happens):\")
print(df_clean[scale_cols].agg([\"min\", \"max\"]).T.assign(span=lambda d: d[\"max\"] - d[\"min\"]).round(2).to_string())
""")

md("""
Notice `vehicles_involved` and `speed_limit_kmph` are already in small, comparable units — scaling them changes little.
`accidents` needs it. `year` is the interesting one: it spans 2005–2023, and standardising it turns "a calendar year"
into "years relative to the mean", which is at least interpretable, though for a trend you would rather engineer
`years_since_2005`.

### Skew correction before scaling
""")

code("""
print(\"skew before log:\", df_clean[\"accidents\"].skew().round(2))
log_acc = np.log1p(df_clean[\"accidents\"])
print(\"skew after log :\", log_acc.skew().round(2), \" <- |skew| < 0.5 is roughly symmetric\")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].hist(df_clean[\"accidents\"], bins=45, color=\"#c44e52\"); axes[0].set_title(\"raw accidents (right-skewed)\")
axes[1].hist(log_acc, bins=45, color=\"#55a868\"); axes[1].set_title(\"log1p(accidents) (near-symmetric)\")
plt.tight_layout(); plt.show()
""")

md("""
**Step 4 checklist** ✅ know your model (needs scaling or not) → choose the scaler by skew/outliers →
log-transform heavy tails → **fit on train, transform test**.
""")

# ---------------------------------------------------------------- STEP 5
md("""
---
## Step 5 — Visualise outliers with boxplots, then remove them

### What a boxplot actually draws

```
        |---- whisker = Q1 - 1.5*IQR  (or min if smaller)
   +----+----+
   |    |    |      box = Q1 .. Q3   (the middle 50% of the data)
   |    |    |      line = median
   +----+----+
        |---- whisker = Q3 + 1.5*IQR
                 o  o   <- points beyond the whiskers = "outliers"
```

Two ways to catch them numerically:

* **IQR rule (Tukey):** outlier if `x < Q1 - 1.5·IQR` or `x > Q3 + 1.5·IQR`. Distribution-free. Use for skewed data.
* **Z-score rule:** outlier if `|x - mean| / std > 3`. Assumes normality — **on skewed data this flags too many legitimate values.**

### The most important warning in this notebook

**An outlier is not automatically an error.** Before deleting anything, ask: *is this a data-entry bug, or is it
the most informative row in the dataset?*

In accident data, a cluster with 25 vehicles involved is probably a real pile-up on a foggy highway. Deleting pile-ups
from a road-safety dataset deletes the very thing you are trying to study. We will remove **impossible** values
(negative ages, `speed_limit = 999`) and **flag but keep** the extreme-but-real ones.
""")

code("""
box_cols = [\"accidents\", \"deaths\", \"injuries\", \"vehicles_involved\", \"speed_limit_kmph\", \"driver_age_median\"]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.ravel(), box_cols):
    sns.boxplot(x=df_clean[col], ax=ax, color=\"#4c72b0\", fliersize=3)
    ax.set_title(f\"{col}   (skew={df_clean[col].skew():.2f})\")
    ax.set_xlabel(\"\")
plt.suptitle(\"Boxplots BEFORE cleaning - note the sentinel spikes at 999 and the long right tails\", y=1.02)
plt.tight_layout(); plt.show()
""")

md("""
Read those six panels like a detective:

* `speed_limit_kmph` — a lone spike at the far right: that's the `999` sentinel code, **not** a fast road.
* `driver_age_median` — dots near 0 and beyond 100: impossible human ages.
* `accidents`, `deaths`, `injuries` — long right tails with no gap. These are **not errors**, they are big states.
  Tukey's rule flags roughly **10% of rows in each** of these columns purely because the distribution is skewed, not
  because anything is wrong. Delete them and you have deleted every large state in India.

### 5a. Quantify the outliers first — never delete blind
""")

code("""
def iqr_report(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((s < lo) | (s > hi)).sum()
    return pd.Series({
        \"Q1\": round(q1, 2), \"Q3\": round(q3, 2), \"IQR\": round(iqr, 2),
        \"lower_fence\": round(lo, 2), \"upper_fence\": round(hi, 2),
        \"n_outliers\": int(n_out), \"pct\": round(n_out / len(s) * 100, 2),
    })

pd.DataFrame({c: iqr_report(df_clean[c]) for c in box_cols}).T
""")

code("""
# Z-score vs IQR on the same column: which rule is right for skewed data?
col = \"accidents\"
z = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
q1, q3 = df_clean[col].quantile([0.25, 0.75])
iqr_flag = (df_clean[col] < q1 - 1.5 * (q3 - q1)) | (df_clean[col] > q3 + 1.5 * (q3 - q1))

print(f\"Z-score |z|>3 flags : {int((z.abs() > 3).sum())} rows\")
print(f\"IQR 1.5x rule flags  : {int(iqr_flag.sum())} rows\")
print(\"\\nBoth flag rows that are perfectly valid - large states simply have more accidents.\")
print(\"Also note the IQR upper fence is 2186, so the 999 sentinel was NOT flagged as an\")
print(\"outlier in that column. Tukey's rule cannot rescue you from implausible values.\")
""")

md("""
### 5b. Remove what is genuinely impossible

Domain knowledge, not statistics, decides these. A speed limit of 999 kmph, an age of -1, a negative injury count:
these are **data-entry errors** and every one of them should go, no matter what a boxplot says.
""")

code("""
before = len(df_clean)

VALID_RANGES = {
    \"speed_limit_kmph\": (20, 120),      # India: 20 kmph on ghat roads up to 120 on expressways
    \"driver_age_median\": (16, 80),      # licence age floor, generous upper bound
    \"vehicles_involved\": (1, 12),       # >12 is a mass-casualty pile-up, treat separately
    \"accidents\": (0, 60_000),
    \"deaths\": (0, 25_000),
    \"injuries\": (0, 60_000),
}
for col, (lo, hi) in VALID_RANGES.items():
    bad = ~df_clean[col].between(lo, hi)
    if bad.any():
        print(f\"{col:20s} dropping {int(bad.sum()):4d} impossible values  (kept range {lo}-{hi})\")
    df_clean = df_clean[~bad]

print(f\"\\n{before - len(df_clean)} rows removed ({(before-len(df_clean))/before:.2%}) -> {len(df_clean)} rows\")
""")

md("""
### 5c. Winsorising — the gentler alternative to deletion

When values are extreme but *real* (a genuine 40-vehicle pile-up), don't delete the row and lose the
other 12 columns with it. **Cap** the value at the 99th percentile instead. The row survives, the extreme
value stops dominating. This is called **winsorising** or **clipping**.
""")

code("""
df_winsor = df_clean.copy()
for col in [\"accidents\", \"deaths\", \"injuries\", \"vehicles_involved\"]:
    lo, hi = df_winsor[col].quantile([0.01, 0.99])
    n_changed = ((df_winsor[col] < lo) | (df_winsor[col] > hi)).sum()
    # .round() keeps these as whole counts - a road-accident cluster cannot be 7650.25 accidents
    df_winsor[col] = df_winsor[col].clip(lo, hi).round()
    print(f\"{col:20s} capped {n_changed:5d} values to [{lo:.0f}, {hi:.0f}]\")

print(\"\\nrows before / after winsorising:\", len(df_clean), \"/\", len(df_winsor), \"(none lost)\")
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(x=df_clean[\"accidents\"], ax=axes[0], color=\"#c44e52\")
axes[0].set_title(f\"BEFORE - accidents (max {df_clean['accidents'].max():,.0f})\")
sns.boxplot(x=df_winsor[\"accidents\"], ax=axes[1], color=\"#55a868\")
axes[1].set_title(f\"AFTER - clipped to 1st/99th percentile (max {df_winsor['accidents'].max():,.0f})\")
plt.suptitle(\"Same rows, no data lost - only the tail is pulled in\", y=1.03)
plt.tight_layout(); plt.show()
""")

code("""
# After cleaning, re-draw every boxplot to confirm
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.ravel(), box_cols):
    sns.boxplot(x=df_winsor[col], ax=ax, color=\"#55a868\", fliersize=3)
    ax.set_title(f\"{col} (cleaned)\")
    ax.set_xlabel(\"\")
plt.suptitle(\"Boxplots AFTER cleaning - impossible codes gone, real tails retained\", y=1.02)
plt.tight_layout(); plt.show()
""")

md("""
**Step 5 checklist** ✅ draw the boxplot → compute the fences → decide *error vs signal* using domain knowledge →
delete impossible values, clip extreme real values → redraw to confirm → never delete based on a threshold alone.
""")

# ---------------------------------------------------------------- WRAP UP
md("""
---
## All five steps in one production pipeline

Everything above was step-by-step for learning. In real code you write it **once**, as a `Pipeline`, so that
every statistic is fitted on `X_train` and merely applied to `X_test`. This is what prevents leakage.

The pipeline below predicts `severity` (Minor / Serious / Fatal) from the circumstances of an accident cluster.
""")

code("""
model_df = df_winsor.copy()

# ---- engineered features (Step 4 bonus): ratios carry more signal than raw counts
model_df[\"deaths_per_accident\"] = model_df[\"deaths\"] / model_df[\"accidents\"].clip(lower=1)
model_df[\"injuries_per_accident\"] = model_df[\"injuries\"] / model_df[\"accidents\"].clip(lower=1)
model_df[\"years_since_2005\"] = model_df[\"year\"] - 2005
model_df[\"is_national_highway\"] = (model_df[\"road_type\"] == \"National Highway\").astype(int)

FEATURES_NUM = [\"speed_limit_kmph\", \"driver_age_median\", \"vehicles_involved\",
                \"years_since_2005\", \"is_national_highway\"]
FEATURES_CAT = [\"road_type\", \"weather\", \"vehicle_type\", \"accident_cause\"]
TARGET = \"severity\"

X = model_df[FEATURES_NUM + FEATURES_CAT]
y = model_df[TARGET]
print(\"X:\", X.shape, \"| y:\", y.shape)
print(y.value_counts().to_string())
""")

code("""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f\"train {len(X_train):,} rows | test {len(X_test):,} rows\")

preprocess = ColumnTransformer(
    transformers=[
        # numeric: median-impute (robust to skew) then standardise
        (\"num\", Pipeline([
            (\"impute\", SimpleImputer(strategy=\"median\")),
            (\"scale\", StandardScaler()),
        ]), FEATURES_NUM),
        # categorical: most-frequent fill then one-hot, ignoring unseen categories
        (\"cat\", Pipeline([
            (\"impute\", SimpleImputer(strategy=\"most_frequent\")),
            (\"encode\", OneHotEncoder(handle_unknown=\"ignore\", drop=\"first\")),
        ]), FEATURES_CAT),
    ]
)
print(\"preprocessor built: impute -> scale (num) | impute -> one-hot (cat)\")
""")

code("""
results = {}
models = {
    \"Logistic Regression (needs scaling)\": LogisticRegression(max_iter=1000, class_weight=\"balanced\"),
    \"Random Forest (scaling irrelevant)\": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
}
for name, clf in models.items():
    pipe = Pipeline([(\"prep\", preprocess), (\"model\", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = accuracy_score(y_test, pred)
    print(f\"{name:40s} accuracy = {results[name]:.3f}\")

best = Pipeline([(\"prep\", preprocess), (\"model\", RandomForestClassifier(
    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))]).fit(X_train, y_train)
print(\"\\n\", classification_report(y_test, best.predict(X_test), zero_division=0))
""")

md("""
Both models land at about **0.64 accuracy**, against the 0.48 majority-class baseline. Note that the Random Forest,
which does not need scaling, is no better than the Logistic Regression that does — that is normal. Scaling is about
letting the optimiser work correctly, not about buying accuracy.

Why 0.64 and not 0.95? Because `severity` here is largely a function of `deaths / accidents`, and I deliberately
**excluded** that ratio from the features. Including it would be **target leakage**: a beautiful score in the notebook
and a useless model in the field, since you do not know the death toll at the moment you are deciding how to respond.
A cleaner process, not a bigger number, is the point of this notebook.

### Persist your work
""")

code("""
import joblib

model_df.to_csv(\"data/india_road_accidents_clean.csv\", index=False)
joblib.dump(best, \"data/severity_model.joblib\")

print(\"clean dataset :\", model_df.shape)
print(\"missing values:\", int(model_df.isna().sum().sum()))
print(\"saved         : data/india_road_accidents_clean.csv, data/severity_model.joblib\")
""")

md("""
### Cheat sheet — the whole notebook on one page

```
1. PROFILE       df.shape, df.info(), df.dtypes, df.describe(), df.nunique(), df.duplicated().sum()

2. MISSING       df.replace(["NA","N/A","-","","Unknown","null"], np.nan)          <- do this FIRST
                 pd.to_numeric(..., errors="coerce")                               <- fix trapped numbers
                 drop_duplicates(); dropna(subset=[target])                        <- unrepairable rows
                 skewed  -> median (or group-wise median) | symmetric -> mean
                 categorical -> mode, or an explicit "Unknown" level
                 NEVER fit an imputer on the full dataset

3. ENCODE        nominal   -> OneHotEncoder(drop="first", handle_unknown="ignore")
                 ordered   -> OrdinalEncoder(categories=[...]) / explicit mapping
                 high-card -> target encoding inside CV, or label encoding for trees only

4. SCALE         tree models       -> not needed
                 linear/SVM/kNN    -> StandardScaler
                 bounded output    -> MinMaxScaler
                 outliers present  -> RobustScaler
                 heavy skew        -> np.log1p FIRST, then scale
                 fit on X_train, transform X_test

5. OUTLIERS      boxplot -> Q1, Q3, IQR -> fences at Q1-1.5*IQR / Q3+1.5*IQR
                 impossible (age<0, speed=999) -> DELETE
                 extreme but real              -> CLIP (winsorise) at 1st/99th pct
                 ask "error or signal?" BEFORE deleting anything
```

### Where to go next

* **Real data:** MoRTH publishes "Road Accidents in India" every year (data.gov.in / morth.nic.in);
  NCRB's *Accidental Deaths & Suicides in India* has state-level tables. Try cleaning the real PDF-to-CSV export —
  it will be far messier than this file.
* **Try next:** `Feature-engine` or `sklearn.compose.make_column_selector` for less boilerplate;
  `IterativeImputer` for MICE; `KNNImputer` on correlated columns.
* **Go deeper on leakage:** wrap the whole pipeline in `cross_val_score(pipe, X, y, cv=5)` —
  because the pipeline is self-contained, sklearn refits the imputer and scaler inside every fold for you.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
out = ROOT / "clean_and_prepare.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
