import pandas as pd
from .config import SKU_UNIT_VALUE, MARGIN_RATE

def compute_metrics(inv_states: pd.DataFrame, transfers: pd.DataFrame, lanes: pd.DataFrame):
    # fill rate
    total_demand = inv_states["demand"].sum()
    total_fulfilled = inv_states["fulfilled"].sum()
    fill_rate = (total_fulfilled / total_demand) if total_demand > 0 else 1.0

    # lost sales cost proxy
    inv_states["lost_sales_cost"] = inv_states.apply(lambda r: r["lost_sales_units"] * (SKU_UNIT_VALUE[r["sku"]] * MARGIN_RATE), axis=1)
    lost_sales_cost = inv_states["lost_sales_cost"].sum()

    # transfer cost
    if transfers is None or transfers.empty:
        transfer_cost = 0.0
    else:
        t = transfers.merge(lanes, on=["from_wh","to_wh"], how="left")
        transfer_cost = (t["qty"] * t["cost_per_unit"]).sum()

    return {
        "fill_rate": float(fill_rate),
        "lost_sales_cost": float(lost_sales_cost),
        "transfer_cost": float(transfer_cost),
        "net_benefit_proxy": float(-transfer_cost - lost_sales_cost)  # compare between policies
    }
