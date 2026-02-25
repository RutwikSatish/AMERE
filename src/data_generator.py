import numpy as np
import pandas as pd
from .config import WAREHOUSES, SKUS, SKU_UNIT_VALUE, BASE_DEMAND, WH_MULT

def build_sku_df():
    return pd.DataFrame([{
        "sku": s,
        "unit_value": SKU_UNIT_VALUE[s],
        "base_demand": BASE_DEMAND[s],
    } for s in SKUS])

def build_lanes_df():
    # fully-connected lanes, symmetric
    rows = []
    for a in WAREHOUSES:
        for b in WAREHOUSES:
            if a == b: 
                continue
            # simple distance proxy by index difference
            ia, ib = WAREHOUSES.index(a), WAREHOUSES.index(b)
            d = abs(ia - ib)
            lead = 1 if d == 1 else 2 if d == 2 else 3
            cost = 0.40 if lead == 1 else 0.70 if lead == 2 else 1.10
            rows.append({"from_wh": a, "to_wh": b, "lead_days": lead, "cost_per_unit": cost, "cap_units": 10**9})
    return pd.DataFrame(rows)

def generate_demand(days: int, seed: int = 7, scenario: str = "viral_spike"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    
    # shock definitions
    shocks = []
    if scenario == "normal":
        shocks = []
    elif scenario == "viral_spike":
        shocks = [{"start":25, "end":30, "wh":"W4_East", "sku":"S3", "mult":3.0, "type":"viral"}]
    elif scenario == "multi_shock":
        shocks = [
            {"start":55, "end":60, "wh":"W1_West", "sku":"S1", "mult":2.0, "type":"promo"},
            {"start":70, "end":75, "wh":"W3_South", "sku":"S5", "mult":2.5, "type":"viral"},
            {"start":72, "end":78, "wh":"W2_Central", "sku":"S2", "mult":2.0, "type":"viral"},
        ]
    else:
        raise ValueError("scenario must be: normal, viral_spike, multi_shock")
    
    for t, date in enumerate(dates, start=1):
        for wh in WAREHOUSES:
            for sku in SKUS:
                base = BASE_DEMAND[sku] * WH_MULT[wh]
                noise = rng.uniform(0.9, 1.1)
                mult = 1.0
                shock_flag = 0
                shock_type = ""
                for sh in shocks:
                    if sh["start"] <= t <= sh["end"] and sh["wh"] == wh and sh["sku"] == sku:
                        mult *= sh["mult"]
                        shock_flag = 1
                        shock_type = sh["type"]
                lam = max(base * noise * mult, 0.1)
                dem = rng.poisson(lam)
                rows.append({"date": date, "day": t, "wh": wh, "sku": sku, "demand": int(dem),
                             "shock_flag": shock_flag, "shock_type": shock_type})
    return pd.DataFrame(rows)
