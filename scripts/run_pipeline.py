"""
run_pipeline.py — the five requested steps, executed in order on the raw accident data.

  1. Import the dataset and explore basic info (shape, dtypes, n unique values)
  2. Handle missing values using mean / median / imputation
  3. Convert categorical features into numerical using encoding
  4. Normalize / standardize the numerical features
  5. Visualize outliers using boxplots and remove them

Outputs
  output/figs/*.png              figures for each step
  output/summary.json            every number quoted in the report
  output/run_log.txt             full console log
  data/road_accidents_clean_final.csv   final modelling-ready dataset

Run:  python3 scripts/run_pipeline.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 40)

RAW = "india_road_accidents_raw.csv"
MISSING_TOKENS = ["NA", "N/A", "", "Unknown", "null", "-", "nan", "None"]
NUM_COLS = ["accidents", "deaths", "injuries", "vehicles_involved",
            "speed_limit_kmph", "driver_age_median"]
NOMINAL = ["road_type", "weather", "vehicle_type", "accident_cause", "alcohol_involved"]
SEVERITY_ORDER = {"Minor": 0, "Serious": 1, "Fatal": 2}

summary = {}
log_lines = []


def say(title=""):
    line = f"\n{'=' * 78}\n{title}\n{'=' * 78}" if title else ""
    print(line)
    log_lines.append(line)


def note(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    log_lines.append(text)


def savefig(fig, name):
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    note(f"    -> saved {path.relative_to(ROOT)}")
    return str(path.relative_to(ROOT))


# ===========================================================================
say("STEP 1 — IMPORT THE DATASET AND EXPLORE BASIC INFO")
# ===========================================================================
df_raw = pd.read_csv(DATA / RAW)
df = df_raw.copy()

note(f"Loaded data/{RAW}")
note(f"  rows x columns : {df.shape[0]:,} x {df.shape[1]}")
note(f"  memory         : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
note("\nPer-column profile:")
profile = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "n_unique": df.nunique(),
    "example": [str(df[c].dropna().iloc[0])[:22] for c in df.columns],
})
note(profile.to_string())

n_dup = int(df.duplicated().sum())
note(f"\nDuplicate rows: {n_dup}")

textly_numeric = [c for c in NUM_COLS if df[c].dtype == object]
note(f"\ndtype check: {len(textly_numeric)} of {len(NUM_COLS)} numeric-looking columns "
     f"are stored as text -> {textly_numeric}")

coerced = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
note("\ndescribe() after coercion — note the impossible min/max values:")
note(coerced.describe().round(2).to_string())

nun = {c: int(df[c].nunique()) for c in ["state", "road_type", "vehicle_type", "weather"]}
note(f"\nCategory counts BEFORE cleaning: {nun}")

summary["step1"] = {
    "rows": int(df.shape[0]), "cols": int(df.shape[1]),
    "duplicates": n_dup, "textly_numeric": textly_numeric,
    "n_unique_before": nun,
    "describe_before": json.loads(coerced.describe().round(2).to_json()),
}

# ---------------------------------------------------------------------------
say("STEP 2 — HANDLE MISSING VALUES (mean / median / imputation)")
# ---------------------------------------------------------------------------
# 2a. one representation of "missing"
df = df.replace(MISSING_TOKENS, np.nan)
df = df.apply(lambda s: s.str.strip() if s.dtype == object else s)

miss = df.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
note("Missing values after normalising the NA tokens:")
note(pd.DataFrame({"missing": miss, "percent": (miss / len(df) * 100).round(2)}).to_string())
summary["step2_missing_pct"] = {k: round(v, 2) for k, v in (miss / len(df) * 100).items()}

# figure: missingness
fig, ax = plt.subplots(figsize=(8, 3.6))
(miss / len(df) * 100).plot.barh(ax=ax, color="#c44e52")
ax.set_xlabel("% missing")
ax.set_title("Step 2 — missing values per column (after normalising NA tokens)")
ax.invert_yaxis()
summary["fig_missing"] = savefig(fig, "step2_missing.png")

# 2b. fix dtypes — numbers trapped in text
for col in NUM_COLS:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

# 2c. drop rows that cannot be repaired
before = len(df)
df = df.drop_duplicates()
df = df.dropna(subset=["severity", "accidents", "year", "state"])
note(f"\nDropped {before - len(df)} unrepairable rows (duplicates / missing target) "
     f"-> {len(df):,} rows")

# 2d. mean vs median — which one fits this data?
stats = pd.DataFrame({
    "mean": df[NUM_COLS].mean(), "median": df[NUM_COLS].median(), "skew": df[NUM_COLS].skew(),
})
stats["gap_%"] = ((stats["mean"] - stats["median"]).abs() / stats["median"] * 100).round(1)
note("\nMean vs median (the gap tells you the distribution is skewed):")
note(stats.round(2).to_string())
summary["step2_stats"] = json.loads(stats.round(2).to_json())

# 2e. compare three imputation methods on one column
col = "driver_age_median"
work = df[[col]].copy()
mean_imp = work[col].fillna(work[col].mean())
median_imp = work[col].fillna(work[col].median())
from sklearn.impute import KNNImputer
knn_imp = pd.Series(KNNImputer(n_neighbors=5).fit_transform(work)[:, 0], index=work.index)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), sharex=True)
for ax, (name, s) in zip(axes, [("mean", mean_imp), ("median", median_imp), ("KNN-5", knn_imp)]):
    ax.hist(s, bins=35, color="#4c72b0")
    ax.set_title(f"{name}  (median={s.median():.1f}, sd={s.std():.2f})")
    ax.set_xlabel(col)
fig.suptitle("Step 2 — three imputation methods on the same column", y=1.04)
fig.tight_layout()
summary["fig_impute"] = savefig(fig, "step2_imputation_methods.png")
note(f"\nVariance: raw={work[col].var():.2f} | mean={mean_imp.var():.2f} | "
     f"median={median_imp.var():.2f} | KNN={knn_imp.var():.2f}")

# 2f. apply the chosen imputations
#   counts are right-skewed -> median;  driver age -> median within its own vehicle group
df["driver_age_median"] = df.groupby("vehicle_type")["driver_age_median"].transform(
    lambda s: s.fillna(s.median()))
df["driver_age_median"] = df["driver_age_median"].fillna(df["driver_age_median"].median())

for c in ["accidents", "deaths", "injuries", "vehicles_involved", "speed_limit_kmph"]:
    df[c] = df[c].fillna(df[c].median())

for c in ["weather", "accident_cause", "alcohol_involved"]:
    df[c] = df[c].fillna(df[c].mode()[0])

remaining = int(df.isna().sum().sum())
note(f"\nImputation applied: median (skewed counts), group-wise median (driver age), "
     f"mode (categoricals)")
note(f"Missing values remaining: {remaining}")
assert remaining == 0, "imputation failed"
summary["step2_remaining_missing"] = remaining
summary["step2_rows_after"] = len(df)
summary["step2_rows_dropped"] = before - len(df)

# ---------------------------------------------------------------------------
say("STEP 3 — CONVERT CATEGORICAL FEATURES INTO NUMERICAL (ENCODING)")
# ---------------------------------------------------------------------------
def tidy(series):
    return series.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


CANONICAL = {
    "vehicle_type": {"two wheeler": "Two-Wheeler", "two-wheeler": "Two-Wheeler",
                     "car/taxi/van": "Car / Taxi / Van", "car / taxi / van": "Car / Taxi / Van",
                     "auto-rickshaw": "Auto-Rickshaw", "truck / lorry": "Truck / Lorry",
                     "bus": "Bus", "pedestrian / cyclist": "Pedestrian / Cyclist"},
    "road_type": {"national highway": "National Highway", "state highway": "State Highway",
                  "other roads": "Other Roads"},
    "state": {"jammu & kashmir": "Jammu And Kashmir", "other uts": "Other UTs"},
}

# count the variants as they appear in the ORIGINAL file (NA tokens excluded),
# otherwise Step 2's whitespace-stripping would hide the mess we are about to fix
CAT_TO_AUDIT = ["state", "road_type", "vehicle_type", "weather", "accident_cause"]
raw_variants = {c: int(df_raw[c].replace(MISSING_TOKENS, np.nan).nunique()) for c in CAT_TO_AUDIT}

for c in ["state", "road_type", "vehicle_type", "weather", "accident_cause", "alcohol_involved"]:
    df[c] = tidy(df[c])
    df[c] = df[c].replace(CANONICAL.get(c, {})).str.title()

clean_variants = {c: int(df[c].nunique()) for c in CAT_TO_AUDIT}
note("Category counts, before -> after merging variant spellings:")
for c in CAT_TO_AUDIT:
    note(f"  {c:16s} {raw_variants[c]:3d} -> {clean_variants[c]:3d}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
for ax, (c, title) in zip(axes, [("vehicle_type", "before"), ("vehicle_type", "after")]):
    src = df_raw["vehicle_type"] if title == "before" else df["vehicle_type"]
    src.value_counts().plot.barh(ax=ax, color="#c44e52" if title == "before" else "#55a868")
    ax.set_title(f"vehicle_type — {title} cleaning")
    ax.invert_yaxis()
fig.tight_layout()
summary["fig_categories"] = savefig(fig, "step3_categories.png")

# 3a. ordinal encoding (features with a real order)
df["severity_code"] = df["severity"].map(SEVERITY_ORDER)
note(f"\nOrdinal encoding of `severity` (the target): {SEVERITY_ORDER}")
note(df["severity_code"].value_counts(normalize=True).sort_index().round(3).to_string())

# 3b. one-hot encoding (nominal features)
cols_before = df.shape[1]
df = pd.get_dummies(df, columns=NOMINAL, drop_first=True, dtype=int)
df = df.loc[:, ~df.columns.duplicated()]
dummy_cols = [c for c in df.columns if any(c.startswith(n + "_") for n in NOMINAL)]
note(f"\nOne-hot encoding of {NOMINAL}")
note(f"  columns: {cols_before} -> {df.shape[1]}  ({len(dummy_cols)} dummy columns created)")
note(f"  {dummy_cols}")

# 3c. label encoding for the high-cardinality state column
df["state_code"] = df["state"].astype("category").cat.codes
note(f"\nLabel encoding `state` ({df['state'].nunique()} categories) -> `state_code` "
     f"(integer, for tree models only)")
note(f"  {dict(df[['state','state_code']].drop_duplicates().sort_values('state_code').head(4).values)}")

summary["step3"] = {
    "raw_variants": raw_variants,
    "clean_variants": clean_variants,
    "columns_after_encoding": int(df.shape[1]),
    "dummy_columns": dummy_cols,
    "severity_map": SEVERITY_ORDER,
}

# ---------------------------------------------------------------------------
say("STEP 4 — NORMALIZE / STANDARDIZE THE NUMERICAL FEATURES")
# ---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scale_cols = NUM_COLS + ["year"]
target_col = "accidents"

comparison = pd.DataFrame({
    "raw": df[target_col],
    "standard": StandardScaler().fit_transform(df[[target_col]]).ravel(),
    "minmax": MinMaxScaler().fit_transform(df[[target_col]]).ravel(),
    "robust": RobustScaler().fit_transform(df[[target_col]]).ravel(),
    "log1p": np.log1p(df[target_col]),
})
note(f"Four representations of `{target_col}`:")
note(comparison.describe().round(3).to_string())
note(f"\nskew raw    : {df[target_col].skew():+.2f}")
note(f"skew log1p  : {np.log1p(df[target_col]).skew():+.2f}   <- only log changes the SHAPE, "
     f"the scalers only relabel the axis")

fig, axes = plt.subplots(1, 5, figsize=(17, 3.3))
for ax, c in zip(axes.ravel(), comparison.columns):
    ax.hist(comparison[c], bins=40, color="#55a868")
    ax.set_title(f"{c}\nmean={comparison[c].mean():.2f}  sd={comparison[c].std():.2f}", fontsize=9.5)
    ax.set_yticks([])
fig.suptitle("Step 4 — one column, five representations: the scalers only relabel the axis, "
             "log1p changes the shape", y=1.04)
fig.tight_layout()
summary["fig_scaling"] = savefig(fig, "step4_scaling.png")

# why it matters: distances are unit-blind
pair = df[["accidents", "vehicles_involved"]].copy()
d_raw = float(np.linalg.norm(pair.iloc[0] - pair.iloc[1]))
d_scaled = float(np.linalg.norm(StandardScaler().fit_transform(pair)[0]
                                - StandardScaler().fit_transform(pair)[1]))
note(f"\nDistance between two rows, raw units    : {d_raw:9.2f}  (dominated by `accidents`)")
note(f"Distance between the same rows, scaled  : {d_scaled:9.2f}  (both features count)")

ranges = df[scale_cols].agg(["min", "max"]).T
ranges["span"] = ranges["max"] - ranges["min"]
note("\nFeature ranges — the reason scaling is needed:")
note(ranges.round(2).to_string())

# apply standardisation (kept as *_scaled columns; recomputed after Step 5)
for c in scale_cols:
    df[f"{c}_scaled"] = StandardScaler().fit_transform(df[[c]])

summary["step4"] = {
    "raw_distance": round(d_raw, 2), "scaled_distance": round(d_scaled, 2),
    "skew_before": round(float(df[target_col].skew()), 2),
    "skew_after_log": round(float(np.log1p(df[target_col]).skew()), 2),
    "ranges": json.loads(ranges.round(2).to_json()),
    "scaled_columns": [f"{c}_scaled" for c in scale_cols],
}
note(f"\nCreated {len(scale_cols)} standardised columns: {[f'{c}_scaled' for c in scale_cols]}")

# ---------------------------------------------------------------------------
say("STEP 5 — VISUALIZE OUTLIERS USING BOXPLOTS AND REMOVE THEM")
# ---------------------------------------------------------------------------
box_cols = NUM_COLS
n_rows_before = len(df)

fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
for ax, c in zip(axes.ravel(), box_cols):
    sns.boxplot(x=df[c], ax=ax, color="#4c72b0", fliersize=2.5)
    ax.set_title(f"{c}  (skew={df[c].skew():.2f})")
    ax.set_xlabel("")
fig.suptitle("Step 5 — boxplots BEFORE removing outliers", y=1.02)
fig.tight_layout()
summary["fig_box_before"] = savefig(fig, "step5_boxplots_before.png")

# quantify before touching anything
def iqr_report(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < lo) | (s > hi)
    return pd.Series({"Q1": q1, "Q3": q3, "IQR": iqr, "lower": lo, "upper": hi,
                      "n_outliers": int(mask.sum()), "pct": round(mask.mean() * 100, 2)})


fences = pd.DataFrame({c: iqr_report(df[c]) for c in box_cols}).T
note("IQR (Tukey) fences per column:")
note(fences.round(2).to_string())

z = (df["accidents"] - df["accidents"].mean()) / df["accidents"].std()
n_z = int((z.abs() > 3).sum())
n_iqr = int(fences.loc["accidents", "n_outliers"])
note(f"\naccidents: z-score |z|>3 flags {n_z} rows, IQR rule flags {n_iqr} rows")
note("  -> both flag valid large states; the rule alone cannot decide what is WRONG")

# 5b. remove impossible values (domain rules, not statistics)
VALID_RANGES = {
    "speed_limit_kmph": (20, 120),   # 20 kmph ghat roads .. 120 expressways
    "driver_age_median": (16, 80),   # licence age floor
    "vehicles_involved": (1, 12),    # >12 = mass-casualty pile-up, treated separately
}
removed = {}
for c, (lo, hi) in VALID_RANGES.items():
    bad = ~df[c].between(lo, hi)
    removed[c] = int(bad.sum())
    df = df[~bad]
    note(f"  {c:20s} removed {removed[c]:4d} impossible values (valid range {lo}-{hi})")

# 5c. clip the extreme-but-real tail instead of deleting rows
clipped = {}
for c in ["accidents", "deaths", "injuries", "vehicles_involved"]:
    lo, hi = df[c].quantile([0.01, 0.99])
    n_changed = int(((df[c] < lo) | (df[c] > hi)).sum())
    df[c] = df[c].clip(lo, hi).round()
    clipped[c] = {"n": n_changed, "lo": round(float(lo), 2), "hi": round(float(hi), 2)}
    note(f"  {c:20s} clipped {n_changed:5d} values to [{lo:.0f}, {hi:.0f}]")

note(f"\nRows: {n_rows_before:,} -> {len(df):,}  "
     f"({n_rows_before - len(df)} removed, {(n_rows_before - len(df)) / n_rows_before:.2%})")

fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
for ax, c in zip(axes.ravel(), box_cols):
    sns.boxplot(x=df[c], ax=ax, color="#55a868", fliersize=2.5)
    if c in ("accidents", "deaths", "injuries"):
        ax.set_xscale("log")   # log axis so the whole tail is readable, not just the box
        ax.set_title(f"{c}  (cleaned, log scale)")
    else:
        ax.set_title(f"{c}  (cleaned)")
    ax.set_xlabel("")
fig.suptitle("Step 5 — boxplots AFTER cleaning (impossible values removed, real tails clipped)", y=1.02)
fig.tight_layout()
summary["fig_box_after"] = savefig(fig, "step5_boxplots_after.png")

# --- headline chart: what exactly happened to each column
fig, ax = plt.subplots(figsize=(9, 3.4))
labels = box_cols
n_removed = [removed.get(c, 0) for c in labels]
n_clipped = [clipped.get(c, {"n": 0})["n"] for c in labels]
x = np.arange(len(labels))
ax.bar(x - 0.2, n_removed, 0.4, label="rows removed (impossible values)", color="#c44e52")
ax.bar(x + 0.2, n_clipped, 0.4, label="values clipped (extreme but real)", color="#4c72b0")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
ax.set_ylabel("count")
ax.set_title("Step 5 — what was removed vs what was clipped, per column")
ax.legend(fontsize=8.5)
fig.tight_layout()
summary["fig_removal"] = savefig(fig, "step5_removed_vs_clipped.png")

# before/after on the worst column
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
sns.boxplot(x=df_raw["accidents"].pipe(lambda s: pd.to_numeric(
    s.astype(str).str.replace(",", "", regex=False), errors="coerce")),
    ax=axes[0], color="#c44e52")
axes[0].set_title("accidents — raw file")
sns.boxplot(x=df["accidents"], ax=axes[1], color="#55a868")
axes[1].set_title("accidents — after cleaning")
fig.suptitle("Step 5 — before / after on the most affected column", y=1.05)
fig.tight_layout()
summary["fig_box_pair"] = savefig(fig, "step5_before_after.png")

# recompute the scalers now that outliers are gone (Step 4 was fitted pre-removal)
for c in scale_cols:
    df[f"{c}_scaled"] = StandardScaler().fit_transform(df[[c]])
note("Recomputed the standardised columns after outlier removal, so the scaling "
     "statistics describe the final data.")

summary["step5"] = {
    "fences": json.loads(fences.round(2).to_json()),
    "zscore_flag": n_z, "iqr_flag": n_iqr,
    "removed_impossible": removed, "clipped": clipped,
    "rows_before": n_rows_before, "rows_after": len(df),
}

# ---------------------------------------------------------------------------
say("FINAL DATASET")
# ---------------------------------------------------------------------------
final_path = DATA / "road_accidents_clean_final.csv"
df.to_csv(final_path, index=False)
note(f"Saved {final_path.relative_to(ROOT)}  ->  {df.shape[0]:,} rows x {df.shape[1]} columns")
note(f"Missing values: {int(df.isna().sum().sum())}")
note(f"Duplicate rows: {int(df.duplicated().sum())}")
note("\nFinal dtypes:")
note(df.dtypes.astype(str).to_string())

summary["final"] = {
    "path": str(final_path.relative_to(ROOT)),
    "rows": int(df.shape[0]), "cols": int(df.shape[1]),
    "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum()),
    "columns": df.columns.tolist(),
}

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
(OUT / "run_log.txt").write_text("\n".join(log_lines))
print(f"\nwrote {OUT/'summary.json'} and {OUT/'run_log.txt'}")
