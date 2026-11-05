"""
Build a realistic (and deliberately MESSY) Indian road-accident dataset for the
data-cleaning tutorial.

Base figures are the official MoRTH / NCRB "Road Accidents in India" series so the
yearly totals in the clean file match the published national numbers.
Grain of one row = (state, year, road_type, vehicle_type) accident cluster.

Run:  python3 scripts/01_make_dataset.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Official national series (MoRTH "Road Accidents in India" / NCRB ADSI)
# ---------------------------------------------------------------------------
OFFICIAL = {
    # year: (accidents, deaths, injuries)
    2005: (439255, 94968, 465282),
    2006: (460920, 105749, 496481),
    2007: (479216, 114590, 513340),
    2008: (484704, 119860, 523193),
    2009: (486384, 126896, 515458),
    2010: (499628, 134513, 527512),
    2011: (497686, 142485, 511394),
    2012: (490383, 138258, 509667),
    2013: (486476, 137572, 494893),
    2014: (489400, 139671, 493474),
    2015: (501423, 146133, 500279),
    2016: (480652, 150785, 494624),
    2017: (464910, 147913, 470975),
    2018: (467044, 151417, 469418),
    2019: (449002, 151113, 451361),
    2020: (366138, 131714, 348279),   # COVID lockdown year
    2021: (412432, 153972, 384448),
    2022: (461312, 168491, 443366),
    2023: (480583, 172890, 462825),
}

STATES = {
    # name: (share of accidents, fatality multiplier)
    "Tamil Nadu": (0.140, 1.25),
    "Madhya Pradesh": (0.110, 1.30),
    "Uttar Pradesh": (0.095, 1.45),
    "Karnataka": (0.090, 1.10),
    "Maharashtra": (0.085, 1.15),
    "Rajasthan": (0.055, 1.30),
    "Andhra Pradesh": (0.050, 1.20),
    "Gujarat": (0.045, 1.00),
    "Telangana": (0.040, 1.05),
    "Kerala": (0.038, 0.85),
    "Bihar": (0.030, 1.35),
    "Odisha": (0.028, 1.25),
    "Punjab": (0.025, 1.10),
    "Haryana": (0.025, 1.15),
    "Chhattisgarh": (0.022, 1.30),
    "Jharkhand": (0.018, 1.20),
    "West Bengal": (0.018, 1.05),
    "Delhi": (0.016, 0.75),
    "Assam": (0.014, 1.25),
    "Uttarakhand": (0.010, 1.20),
    "Himachal Pradesh": (0.009, 1.15),
    "Jammu & Kashmir": (0.008, 1.10),
    "Goa": (0.008, 0.95),
    "Tripura": (0.005, 1.15),
    "Meghalaya": (0.004, 1.20),
    "Manipur": (0.004, 1.15),
    "Nagaland": (0.003, 1.20),
    "Arunachal Pradesh": (0.003, 1.25),
    "Mizoram": (0.002, 1.10),
    "Other UTs": (0.006, 0.90),
}

ROAD_TYPES = {"National Highway": 0.31, "State Highway": 0.22, "Other Roads": 0.47}
ROAD_TYPE_FATAL = {"National Highway": 1.20, "State Highway": 1.05, "Other Roads": 0.88}

VEHICLES = {
    # vehicle_type: (share of accidents, fatality multiplier)
    "Two-wheeler": (0.44, 1.00),
    "Car / Taxi / Van": (0.20, 1.05),
    "Truck / Lorry": (0.11, 1.55),
    "Bus": (0.06, 1.45),
    "Auto-rickshaw": (0.08, 0.95),
    "Pedestrian / Cyclist": (0.11, 1.85),
}
WEATHER = {"Clear": 0.68, "Rainy": 0.16, "Foggy": 0.09, "Hazy / Dusty": 0.07}
CAUSES = {
    "Overspeeding": 0.684,
    "Wrong side / lane indiscipline": 0.053,
    "Drunk driving": 0.019,
    "Mobile phone use": 0.015,
    "Red light jumping": 0.005,
    "Other": 0.224,
}
VEHICLE_AGE = {"Two-wheeler": 30, "Car / Taxi / Van": 42, "Truck / Lorry": 50,
               "Bus": 38, "Auto-rickshaw": 27, "Pedestrian / Cyclist": 36}
SPEED_BY_ROAD = {"National Highway": 100, "State Highway": 80, "Other Roads": 60}


def allocate(total: int, weights: np.ndarray) -> np.ndarray:
    """Split an integer total into integers using the largest-remainder method
    so the parts sum back to exactly `total`."""
    w = weights / weights.sum()
    raw = w * total
    base = np.floor(raw).astype(int)
    short = total - base.sum()
    if short > 0:
        order = np.argsort(-(raw - base))
        base[order[:short]] += 1
    return base


rows = []
for year, (acc_tot, death_tot, injury_tot) in OFFICIAL.items():
    combos = [(s, rt, v) for s in STATES for rt in ROAD_TYPES for v in VEHICLES]
    n = len(combos)
    noise = rng.lognormal(mean=0.0, sigma=0.30, size=n)
    w_acc = np.array([STATES[s][0] * ROAD_TYPES[rt] * VEHICLES[v][0] for s, rt, v in combos]) * noise
    acc = allocate(acc_tot, w_acc)

    # a genuinely bad year (2020) shifts the mix slightly: less NH travel
    w_death = acc * np.array([
        STATES[s][1] * ROAD_TYPE_FATAL[rt] * VEHICLES[v][1] for s, rt, v in combos
    ]) * rng.lognormal(0.0, 0.22, size=n)
    deaths = allocate(death_tot, w_death)

    w_inj = acc * np.array([1.0 + 0.35 * (VEHICLES[v][1] - 1) for s, rt, v in combos]) \
        * rng.lognormal(0.0, 0.25, size=n)
    injuries = allocate(injury_tot, w_inj)

    for i, (s, rt, v) in enumerate(combos):
        fat_ratio = deaths[i] / max(acc[i], 1)
        if fat_ratio > 0.42:
            sev = "Fatal"
        elif fat_ratio > 0.24:
            sev = "Serious"
        else:
            sev = "Minor"
        rows.append({
            "state": s,
            "year": year,
            "road_type": rt,
            "vehicle_type": v,
            "weather": rng.choice(list(WEATHER), p=list(WEATHER.values())),
            "accident_cause": rng.choice(list(CAUSES), p=list(CAUSES.values())),
            "severity": sev,
            "accidents": int(acc[i]),
            "deaths": int(deaths[i]),
            "injuries": int(injuries[i]),
            "vehicles_involved": max(1, int(rng.poisson(2.1 if v != "Pedestrian / Cyclist" else 1.2) + 1)),
            "speed_limit_kmph": int(SPEED_BY_ROAD[rt] + rng.choice([-10, 0, 0, 10])),
            "driver_age_median": float(np.clip(rng.normal(VEHICLE_AGE[v], 5.5), 18, 70)),
            "alcohol_involved": rng.choice(["Yes", "No"], p=[0.045, 0.955]),
        })

df = pd.DataFrame(rows)
print(f"clean base rows: {len(df):,}")
for col, tot in [("accidents", 0), ("deaths", 1), ("injuries", 2)]:
    got = df.groupby("year")[col].sum()
    want = pd.Series({y: OFFICIAL[y][tot] for y in OFFICIAL})
    assert (got == want).all(), col
print("yearly totals match the official MoRTH series exactly ✓")

# ---------------------------------------------------------------------------
# 2. Now make it DIRTY, the way real downloaded data looks
# ---------------------------------------------------------------------------
dirty = df.copy()
# numeric columns become 'object' dtype once we sprinkle text tokens into them -
# that is exactly what happens in real life when you read a messy CSV.
for c in ["accidents", "deaths", "injuries", "vehicles_involved",
          "speed_limit_kmph", "driver_age_median"]:
    dirty[c] = dirty[c].astype(object)

# (a) inconsistent category spellings / stray whitespace / case
idx = dirty.index[rng.random(len(dirty)) < 0.06]
dirty.loc[idx, "vehicle_type"] = dirty.loc[idx, "vehicle_type"].replace(
    {"Two-wheeler": "two wheeler", "Car / Taxi / Van": "Car/Taxi/Van"})
idx = dirty.index[rng.random(len(dirty)) < 0.05]
dirty.loc[idx, "road_type"] = dirty.loc[idx, "road_type"] + " "
idx = dirty.index[rng.random(len(dirty)) < 0.04]
dirty.loc[idx, "state"] = dirty.loc[idx, "state"].str.upper()
idx = dirty.index[rng.random(len(dirty)) < 0.05]
dirty.loc[idx, "weather"] = dirty.loc[idx, "weather"].str.lower()

# (b) missing values, written in every messy way analysts actually meet
NA_TOKENS = ["NA", "N/A", "", "Unknown", "-", "null"]
def sprinkle(col, frac, tokens=NA_TOKENS):
    pos = dirty.index[rng.random(len(dirty)) < frac]
    dirty.loc[pos, col] = rng.choice(tokens, size=len(pos))
    return pos

sprinkle("weather", 0.09)
sprinkle("accident_cause", 0.11)
sprinkle("alcohol_involved", 0.07, ["NA", "", "Unknown"])
sprinkle("driver_age_median", 0.10)
sprinkle("speed_limit_kmph", 0.08)
sprinkle("injuries", 0.05, ["NA", "-", ""])
sprinkle("vehicles_involved", 0.04, ["", "NA"])

# (c) sentinel / impossible codes hiding inside numeric columns
pos = dirty.index[rng.random(len(dirty)) < 0.02]
dirty.loc[pos, "speed_limit_kmph"] = 999
pos = dirty.index[rng.random(len(dirty)) < 0.015]
dirty.loc[pos, "driver_age_median"] = rng.choice([-1, 0, 999, 150], size=len(pos))

# (d) genuine but extreme values (these are the ones a boxplot should catch)
pos = dirty.index[rng.random(len(dirty)) < 0.012]
dirty.loc[pos, "speed_limit_kmph"] = rng.choice([180, 200, 250], size=len(pos))
pos = dirty.index[rng.random(len(dirty)) < 0.010]
dirty.loc[pos, "vehicles_involved"] = rng.choice([18, 25, 31], size=len(pos))
pos = dirty.index[rng.random(len(dirty)) < 0.008]
dirty.loc[pos, "injuries"] = dirty.loc[pos, "injuries"].astype(str)

# (e) numbers stored as text with thousand separators (classic Excel export)
pos = dirty.index[rng.random(len(dirty)) < 0.08]
dirty.loc[pos, "deaths"] = dirty.loc[pos, "deaths"].apply(lambda x: f"{x:,}")
pos = dirty.index[rng.random(len(dirty)) < 0.06]
dirty.loc[pos, "accidents"] = dirty.loc[pos, "accidents"].apply(lambda x: f"{x:,}")

# (f) duplicate rows from a double download
dupes = dirty.sample(frac=0.02, random_state=7)
dirty = pd.concat([dirty, dupes], ignore_index=True).sample(frac=1, random_state=11).reset_index(drop=True)

dirty.to_csv(OUT / "india_road_accidents_raw.csv", index=False)
print(f"dirty file rows: {len(dirty):,}  (incl. {len(dupes)} duplicate rows)")
print(dirty.dtypes.to_string())
