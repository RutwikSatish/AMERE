import os
import sys

# --- Robust path setup for Streamlit Cloud ---
# Case A: src/ is inside the same folder as app.py (amere/src)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

# Case B: src/ is at repo root (src/) while app.py is in amere/
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

import streamlit as st
import pandas as pd

from src.config import PolicyParams
from src.data_generator import build_sku_df, build_lanes_df, generate_demand
from src.simulator import run_simulation
from src.policies.baseline import BaselinePolicy
from src.policies.heuristic_reallocator import HeuristicReallocatorPolicy
from src.metrics import compute_metrics

st.set_page_config(page_title="AMERE – Inventory Reallocation Engine", layout="wide")

st.title("AMERE – Adaptive Multi-Echelon Reallocation Engine")

with st.sidebar:
    st.header("Scenario")
    days = st.selectbox("Days to simulate", [30, 60, 90], index=2)
    scenario = st.selectbox("Shock scenario", ["normal", "viral_spike", "multi_shock"], index=1)

    st.header("Policy")
    policy_name = st.selectbox("Run mode", ["Baseline (no transfers)", "Heuristic Reallocator"], index=1)

    st.header("Heuristic parameters")
    H = st.slider("Forecast horizon (days)", 3, 14, 7)
    buffer_days = st.slider("Buffer days", 0, 5, 2)
    min_move = st.slider("Minimum transfer qty", 1, 25, 5)

    run_btn = st.button("Run Simulation")

if run_btn:
    sku_df = build_sku_df()
    lanes_df = build_lanes_df()
    demand_df = generate_demand(days=days, scenario=scenario)

    if policy_name.startswith("Baseline"):
        policy = BaselinePolicy()
    else:
        params = PolicyParams(horizon_days=H, buffer_days=buffer_days, min_move=min_move)
        policy = HeuristicReallocatorPolicy(params)

    inv_states, transfers = run_simulation(days, demand_df, lanes_df, sku_df, policy)
    m = compute_metrics(inv_states, transfers, lanes_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Fill rate", f"{m['fill_rate']*100:.2f}%")
    c2.metric("Lost sales cost (proxy)", f"${m['lost_sales_cost']:,.0f}")
    c3.metric("Transfer cost", f"${m['transfer_cost']:,.0f}")

    st.subheader("Lost sales over time")
    daily = inv_states.groupby(["day"], as_index=False)[["lost_sales_units", "lost_sales_cost"]].sum()
    st.line_chart(daily.set_index("day")[["lost_sales_units"]])

    st.subheader("Transfer recommendations")
    if transfers is None or transfers.empty:
        st.info("No transfers planned for this run.")
    else:
        st.dataframe(
            transfers.sort_values(["order_day", "sku"]).reset_index(drop=True),
            use_container_width=True
        )

    st.subheader("Inventory state sample")
    st.dataframe(inv_states.head(30), use_container_width=True)
