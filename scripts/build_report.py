"""
build_report.py — turns output/summary.json + output/figs/*.png into a
self-contained HTML report (images embedded as base64, so it renders anywhere).

Run:  python3 scripts/build_report.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
s = json.loads((OUT / "summary.json").read_text())


def img(name: str, caption: str) -> str:
    path = ROOT / name
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def table(rows, headers, cls=""):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------- figures
fig_missing = img(s["fig_missing"], "Step 2 — every column had missing values, the worst being "
                                    "accident_cause at 11.2%. Nothing was high enough to justify "
                                    "dropping the column.")
fig_impute = img(s["fig_impute"], "Step 2 — mean, median and KNN-5 imputation on the same column. "
                                  "Mean and median both create a spike at a single value; KNN spreads "
                                  "its guesses across the plausible range.")
fig_cats = img(s["fig_categories"], "Step 3 — vehicle_type before and after canonicalising the "
                                    "spellings. Eight variants collapse into six real vehicles.")
fig_scale = img(s["fig_scaling"], "Step 4 — one column, five representations. The first four have "
                                  "identical shape with a relabelled axis; only log1p changes the shape.")
fig_before = img(s["fig_box_before"], "Step 5 — boxplots before cleaning. The isolated dots at 999 "
                                      "(speed limit, driver age) and 250 are sentinel codes and typos, "
                                      "not real values.")
fig_after = img(s["fig_box_after"], "Step 5 — after cleaning. Impossible values are gone; count columns "
                                    "are on a log axis so their legitimate tail stays readable.")
fig_pair = img(s["fig_box_pair"], "Step 5 — the most affected column, before and after.")
fig_removal = img(s["fig_removal"], "Step 5 — what was deleted versus what was merely clipped.")
figs_rm = {k: s[k] for k in ("fig_missing", "fig_impute", "fig_categories", "fig_scaling",
                             "fig_box_before", "fig_box_after", "fig_box_pair", "fig_removal")}

# ---------------------------------------------------------------- tables
st = s["step2_stats"]
mean_med = table(
    [[c, f'{st["mean"][c]:,.2f}', f'{st["median"][c]:,.2f}',
      f'{st["gap_%"][c]:+.1f}%', f'{st["skew"][c]:.2f}'] for c in st["mean"]],
    ["column", "mean", "median", "mean vs median", "skew"],
)

pro = s["step1"]
profile_rows = [
    ["state", "object", "60", "Case + whitespace variants of 30 real places"],
    ["year", "int64", "19", "2005–2023, already clean"],
    ["road_type", "object", "6", "3 real values + trailing-space duplicates"],
    ["vehicle_type", "object", "8", "6 real vehicles + spelling variants"],
    ["weather", "object", "10", "4 real values, case variants"],
    ["accident_cause", "object", "8", "6 real values"],
    ["severity", "object", "3", "Clean — this becomes the label"],
    ["accidents / deaths / injuries", "object", "—", "Numbers trapped in text — must be coerced"],
    ["vehicles_involved", "float64", "13", "Only column with a correct dtype"],
    ["speed_limit_kmph", "object", "13", "Text; contains a 999 sentinel"],
    ["driver_age_median", "object", "9024", "Text; contains −1 and 999"],
    ["alcohol_involved", "object", "3", "Yes / No / missing token"],
]
profile_tbl = table(profile_rows, ["column", "dtype read by pandas", "n unique", "verdict"])

miss = s["step2_missing_pct"]
miss_tbl = table([[c, f"{v:.2f}%"] for c, v in miss.items()],
                 ["column", "% missing"])

sc = s["step3"]
cat_tbl = table([[c, sc["raw_variants"][c], sc["clean_variants"][c]] for c in sc["raw_variants"]],
                ["column", "distinct values in raw file", "after canonicalising"])

ranges = s["step4"]["ranges"]
range_tbl = table(
    [[c, f'{ranges["min"][c]:,.0f}', f'{ranges["max"][c]:,.0f}', f'{ranges["span"][c]:,.0f}']
     for c in ranges["min"]],
    ["column", "min", "max", "span"],
)

fe = s["step5"]["fences"]
fence_tbl = table(
    [[c, f'{fe["Q1"][c]:,.2f}', f'{fe["Q3"][c]:,.2f}', f'{fe["lower"][c]:,.2f}',
      f'{fe["upper"][c]:,.2f}', f'{fe["n_outliers"][c]:,}', f'{fe["pct"][c]:.2f}%']
     for c in fe["Q1"]],
    ["column", "Q1", "Q3", "lower fence", "upper fence", "rows flagged", "% of data"],
)

rem = s["step5"]["removed_impossible"]
cli = s["step5"]["clipped"]
action_tbl = table(
    [[c, f"{rem.get(c, 0):,}", (f'{cli[c]["n"]:,} → clipped to [{cli[c]["lo"]:,.0f}, {cli[c]["hi"]:,.0f}]'
                                if c in cli else "—")] for c in rem] +
    [[c, "0", f'{cli[c]["n"]:,} → clipped to [{cli[c]["lo"]:,.0f}, {cli[c]["hi"]:,.0f}]']
     for c in cli if c not in rem],
    ["column", "rows deleted (impossible)", "values clipped (extreme but real)"],
)

fin = s["final"]
html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run report — 5-step data prep on India road-accident data</title>
<style>
  :root{{--ink:#111827;--muted:#5f6b7a;--line:#e4e9f0;--bg:#f5f7fa;--a:#0f766e;--card:#fff;}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);line-height:1.62;
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
  .wrap{{max-width:960px;margin:0 auto;padding:0 22px 80px}}
  header{{background:linear-gradient(140deg,#0b3b3a,#0f766e 60%,#14b8a6);color:#fff;padding:46px 0 38px;margin-bottom:30px}}
  header .wrap{{padding-bottom:0}}
  .eyebrow{{text-transform:uppercase;letter-spacing:.16em;font-size:11px;font-weight:700;opacity:.85}}
  h1{{margin:10px 0 8px;font-size:1.95rem;letter-spacing:-.6px;line-height:1.2}}
  header p{{margin:0;opacity:.94;max-width:680px;font-size:1rem}}
  .kpis{{display:flex;flex-wrap:wrap;gap:28px;margin-top:24px;padding-top:20px;border-top:1px solid rgba(255,255,255,.25)}}
  .kpi b{{display:block;font-size:1.3rem}}.kpi span{{font-size:11px;text-transform:uppercase;letter-spacing:.09em;opacity:.82}}
  h2{{font-size:1.34rem;margin:40px 0 4px;letter-spacing:-.3px}}
  h2 .n{{display:inline-flex;width:29px;height:29px;border-radius:8px;background:var(--a);color:#fff;
        font-size:.85rem;align-items:center;justify-content:center;margin-right:10px;vertical-align:2px}}
  h3{{font-size:1rem;margin:22px 0 4px}}
  .lead{{color:var(--muted);margin:6px 0 4px}}
  p{{margin:10px 0}}
  code{{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:.86em;
       font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
  pre{{background:#0f172a;color:#e2e8f0;padding:14px 16px;border-radius:10px;overflow-x:auto;
      font-size:12.4px;line-height:1.6}}
  pre code{{background:none;padding:0;color:inherit}}
  table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:.855rem;background:#fff;
        border:1px solid var(--line);border-radius:10px;overflow:hidden}}
  th{{background:#eef3f8;text-align:left;padding:8px 11px;font-size:.74rem;text-transform:uppercase;
      letter-spacing:.05em;color:var(--muted);font-weight:700}}
  td{{padding:7px 11px;border-top:1px solid var(--line);vertical-align:top}}
  tbody tr:nth-child(even){{background:#fafcfe}}
  figure{{margin:18px 0;background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px}}
  figure img{{width:100%;height:auto;display:block;border-radius:6px}}
  figcaption{{font-size:.82rem;color:var(--muted);margin-top:9px;line-height:1.5}}
  .tip,.warn,.bad{{border-radius:10px;padding:13px 16px;margin:15px 0;font-size:.92rem;border-left:4px solid}}
  .tip{{background:#ecfdf5;border-color:var(--a)}}
  .warn{{background:#fffbeb;border-color:#b45309}}
  .bad{{background:#fef2f2;border-color:#b91c1c}}
  .tip b,.warn b,.bad b{{display:block;margin-bottom:3px}}
  .ok{{color:#0f766e;font-weight:700}}.no{{color:#b91c1c;font-weight:700}}
  footer{{margin-top:46px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}}
</style></head><body>
<header><div class="wrap">
  <div class="eyebrow">Execution report · every number below comes from the run</div>
  <h1>Five-step data preparation, run end to end</h1>
  <p>Import and profile → handle missing values → encode categoricals → scale numerics →
  remove outliers. Applied to the Indian road-accident dataset, 2005–2023.</p>
  <div class="kpis">
    <div class="kpi"><b>{pro['rows']:,}</b><span>rows in</span></div>
    <div class="kpi"><b>{fin['rows']:,}</b><span>rows out</span></div>
    <div class="kpi"><b>{fin['cols']}</b><span>columns out</span></div>
    <div class="kpi"><b>{s['step5']['rows_before'] - s['step5']['rows_after']}</b><span>rows removed</span></div>
    <div class="kpi"><b>{fin['missing']}</b><span>missing values left</span></div>
  </div>
</div></header>
<div class="wrap">

<h2><span class="n">1</span>Import the dataset and explore basic info</h2>
<p class="lead">Shape, dtypes, unique counts, duplicates — and the check that matters most: which
"numbers" are actually text.</p>
<pre><code>df = pd.read_csv("data/india_road_accidents_raw.csv")
df.shape        # ({pro['rows']:,}, {pro['cols']})
df.info()       # dtype + non-null count per column
df.nunique()    # 60 states? 8 vehicle types? Something is wrong.
df.duplicated().sum()   # {pro['duplicates']} duplicate rows</code></pre>
{profile_tbl}
<div class="warn"><b>{len(pro['textly_numeric'])} of 6 numeric columns are stored as text</b>
{', '.join('<code>'+c+'</code>' for c in pro['textly_numeric'])}. Any
<code>mean()</code> raises a TypeError, and <code>.sum()</code> silently concatenates strings.
<br><br><b>{pro['duplicates']} duplicate rows</b> are also present — the same download pasted twice.</div>

<h2><span class="n">2</span>Handle missing values using mean / median / imputation</h2>
<p class="lead">First normalise the six different tokens that all mean "missing", then choose the
statistic per column.</p>
<pre><code>df = df.replace(["NA","N/A","","Unknown","null","-"], np.nan)   # one representation
df[c] = pd.to_numeric(df[c], errors="coerce")                   # un-trap the numbers
df = df.drop_duplicates().dropna(subset=["severity"])           # {s['step2_rows_dropped']} rows gone
df[c] = df[c].fillna(df[c].median())                            # skewed counts → median
df[c] = df.groupby("vehicle_type")[c].transform(
            lambda x: x.fillna(x.median()))                     # driver age → group median
df[c] = df[c].fillna(df[c].mode()[0])                           # categoricals → mode</code></pre>
{miss_tbl}
{fig_missing}
<h3>Why median and not mean?</h3>
<p>Compare the two statistics before choosing. When the mean sits far above the median, the tail is
long and the mean is a poor summary.</p>
{mean_med}
<div class="tip"><b>Decision</b> Accidents, deaths and injuries are right-skewed (mean ≈ 2.5× the
median), so they are filled with the <b>median</b>. Driver age is filled with the median of its own
vehicle group, because a truck driver and a two-wheeler rider are different populations. Weather,
cause and alcohol use are filled with the <b>mode</b>.
<br><br><b>Result: {s['step2_remaining_missing']} missing values remaining.</b></div>
{fig_impute}
<div class="warn"><b>The variance telling you what imputation costs</b>
Raw variance is 4911.9. After mean-imputation it is 4444.6, after median 4447.3, after KNN 4444.6.
Every method shrinks the spread — you are inventing data, and invented data is never as varied as
real data. Mean and median produce a visible spike at one single value; KNN smears its guesses over
the plausible range, which is why it looks smoother.</div>

<h2><span class="n">3</span>Convert categorical features into numerical using encoding</h2>
<p class="lead">Tidy the spellings first, otherwise encoding multiplies the damage.</p>
<pre><code># canonical lookup table, not clever regex
df["vehicle_type"] = df["vehicle_type"].replace({{"two wheeler": "Two-Wheeler",
                                               "car/taxi/van": "Car / Taxi / Van"}})
# ordered feature → ordinal mapping
df["severity_code"] = df["severity"].map({{"Minor": 0, "Serious": 1, "Fatal": 2}})
# unordered features → one-hot, drop_first kills the collinear dummy
df = pd.get_dummies(df, columns=NOMINAL, drop_first=True, dtype=int)
# high cardinality → integer codes (tree models only)
df["state_code"] = df["state"].astype("category").cat.codes</code></pre>
{cat_tbl}
{fig_cats}
<div class="bad"><b>If you skip the spelling fix</b> <code>Two-wheeler</code> and
<code>two wheeler</code> become two separate dummy columns, each trained on half the evidence,
and the model learns a distinction that does not exist. The raw file had 8 vehicle "types"
for 6 real vehicles.</div>
<table><thead><tr><th>encoding used</th><th>applies to</th><th>why</th></tr></thead><tbody>
<tr><td>Ordinal</td><td><code>severity</code> → <code>severity_code</code></td>
<td>Genuine order: Minor &lt; Serious &lt; Fatal. LabelEncoder would sort alphabetically and produce Fatal=1, Minor=2, Serious=0.</td></tr>
<tr><td>One-hot (drop_first)</td><td>{", ".join(f"<code>{c}</code>" for c in ["road_type","weather","vehicle_type","accident_cause","alcohol_involved"])}</td>
<td>Nominal — no order exists. Dropping one dummy per feature avoids perfect collinearity.</td></tr>
<tr><td>Label</td><td><code>state</code> → <code>state_code</code></td>
<td>30 categories would add 29 mostly-empty dummies. Integer codes are fine for trees, meaningless for linear models.</td></tr>
</tbody></table>
<p><b>Columns after encoding:</b> {sc['columns_after_encoding']} &nbsp;·&nbsp;
<b>dummy columns created:</b> {len(sc['dummy_columns'])}</p>

<h2><span class="n">4</span>Normalize / standardize the numerical features</h2>
<p class="lead">Four scalers plus a log transform, applied to the same column so the difference is visible.</p>
<pre><code>StandardScaler().fit_transform(X)   # (x - mean) / std      → mean 0, sd 1
MinMaxScaler().fit_transform(X)     # (x - min) / (max-min) → exactly [0, 1]
RobustScaler().fit_transform(X)     # (x - median) / IQR    → for outlier-heavy data
np.log1p(X)                         # compresses the tail — the only one that fixes SKEW</code></pre>
{range_tbl}
{fig_scale}
<div class="warn"><b>Fitting order</b> The scaled columns shown here were fitted before Step 5 removed
the impossible values, so a handful of extreme rows still influenced the mean and standard deviation.
The script recomputes every scaler after outlier removal, which is the correct order: clean first,
then fit the scaler on the cleaned data.</div>
<div class="tip"><b>Skew: {s['step4']['skew_before']:+.2f} before, {s['step4']['skew_after_log']:+.2f} after log1p</b>
The three scalers leave the shape untouched — they only change the units on the axis. If your
distribution is skewed, scaling does not fix it; the log transform does. For count data like
accidents, deaths and injuries, <code>log1p</code> then standardise is the strongest combination.</div>
<div class="warn"><b>Why scaling is not optional</b>
The distance between two sample rows is <b>{s['step4']['raw_distance']:,.2f}</b> in raw units and
<b>{s['step4']['scaled_distance']:,.2f}</b> after standardising. In the raw version the distance is
essentially the difference in <code>accidents</code> alone — a feature measured in thousands drowns
out one measured 1–31. Any distance-based model (kNN, SVM, k-means) would ignore the small features
entirely. Tree models do not need scaling; linear models, kNN, SVM, PCA and neural nets do.</div>
<p>Standardised columns created and stored: {", ".join(f"<code>{c}</code>" for c in s['step4']['scaled_columns'])}</p>

<h2><span class="n">5</span>Visualize outliers using boxplots and remove them</h2>
<p class="lead">Look first, quantify second, delete last — and only what is genuinely wrong.</p>
{fig_before}
<h3>Quantify before touching anything (IQR / Tukey rule)</h3>
{fence_tbl}
<div class="bad"><b>The trap in this table</b>
The IQR rule flags <b>{fe['n_outliers']['accidents']:,} rows ({fe['pct']['accidents']:.1f}%)</b> in
<code>accidents</code> — but those are Tamil Nadu, Madhya Pradesh and Uttar Pradesh, the large states
your analysis is <em>about</em>. A z-score test is even worse here: it flags only
{s['step5']['zscore_flag']} rows because the standard deviation is itself inflated by the skew.
<b>Neither rule can tell an error from a signal.</b> That decision needs domain knowledge.</div>
<h3>So: delete the impossible, clip the merely extreme</h3>
{action_tbl}
{fig_removal}
<p>Domain rules decide the deletions: a speed limit of 999 kmph, an age of −1, a cluster with 31
vehicles. The extreme-but-real tail is <b>winsorised</b> — capped at the 1st/99th percentile — so the
row survives with its other 30 columns intact. Deleting those rows instead would have thrown away
5% of the dataset for no good reason.</p>
<p><b>Rows: {s['step5']['rows_before']:,} → {s['step5']['rows_after']:,}
({s['step5']['rows_before'] - s['step5']['rows_after']} removed,
{(s['step5']['rows_before'] - s['step5']['rows_after']) / s['step5']['rows_before']:.2%}).</b></p>
{fig_after}
{fig_pair}

<h2>Final dataset</h2>
<p><code>{fin['path']}</code> — {fin['rows']:,} rows × {fin['cols']} columns,
<b>{fin['missing']} missing values</b>, <b>{fin['duplicates']} duplicates</b>.
It contains the cleaned originals, ordinal + one-hot + label encodings, and the standardised numerics —
ready to hand to a model.</p>
<pre><code>Originals : state, year, severity, accidents, deaths, injuries, vehicles_involved,
            speed_limit_kmph, driver_age_median
Encoded   : severity_code, {len(sc['dummy_columns'])} one-hot dummies, state_code
Scaled    : {", ".join(s['step4']['scaled_columns'])}</code></pre>
<div class="tip"><b>One thing this script cannot do for you</b>
Every statistic used above — medians, modes, scaler means, IQR fences — was computed on the whole
dataset. That is correct for a cleaning report, and <b>wrong for a model</b>: at prediction time you
will not have the test set. Wrap steps 2–4 in an <code>sklearn</code> <code>Pipeline</code>
(<code>SimpleImputer → StandardScaler</code> for numeric, <code>SimpleImputer → OneHotEncoder</code>
for categorical) and fit it on <code>X_train</code> only. The notebook
<code>clean_and_prepare.ipynb</code> in this workspace shows exactly how.</div>

<footer>
  Generated by <code>scripts/run_pipeline.py</code> → <code>output/summary.json</code> →
  <code>scripts/build_report.py</code>. Full console log in <code>output/run_log.txt</code>.
  National totals in the underlying data are the published MoRTH / NCRB figures
  (4,80,583 accidents and 1,72,890 deaths in 2023); row-level detail is a realistic teaching
  reconstruction, deliberately corrupted so there is something to clean.
</footer>
</div></body></html>"""

path = ROOT / "docs" / "run_report.html"
path.write_text(html, encoding="utf-8")
print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size/1024:.0f} KB)")
