import pandas as pd
import numpy as np
from ..config import PolicyParams, SKU_UNIT_VALUE, MARGIN_RATE

LAST_N_DAYS = 7
SHOCK_DAYS = 3

class HeuristicReallocatorPolicy:
    name = "Heuristic Reallocator"

    def __init__(self, params: PolicyParams):
        self.p = params

    def _forecast_demand(self, demand_hist, sku, wh, day):
        recent = demand_hist[(demand_hist["sku"] == sku) & (demand_hist["wh"] == wh) & (demand_hist["day"] <= day)]
        last_n = recent.tail(LAST_N_DAYS)
        if last_n.empty:
            return 0.0
        last_shock = last_n.tail(SHOCK_DAYS)
        if last_shock["shock_flag"].sum() >= 1:
            return float(last_shock["demand"].mean())
        return float(last_n["demand"].mean())

    def _make_lane_lookup(self, lanes_df):
        # Create {(from_wh, to_wh): lane_row}
        return {(row['from_wh'], row['to_wh']): row for _, row in lanes_df.iterrows()}

    def plan_transfers(self, day: int, date, on_hand_after_sales: pd.DataFrame,
                       demand_hist: pd.DataFrame, lanes: pd.DataFrame) -> pd.DataFrame:
        orders = []
        lane_lookup = self._make_lane_lookup(lanes)

        for sku in on_hand_after_sales["sku"].unique():
            inv = on_hand_after_sales[on_hand_after_sales["sku"] == sku].copy()
            # Forecast demand for each wh
            inv["avg_dem"] = inv["wh"].apply(lambda wh: self._forecast_demand(demand_hist, sku, wh, day))
            inv["target"] = (self.p.horizon_days + self.p.buffer_days) * inv["avg_dem"]
            inv["surplus"] = (inv["on_hand"] - inv["target"]).clip(lower=0)
            inv["shortage"] = (inv["target"] - inv["on_hand"]).clip(lower=0)

            donors = inv[inv["surplus"] > 0].copy()
            receivers = inv[inv["shortage"] > 0].copy()

            if donors.empty or receivers.empty:
                continue
            receivers = receivers.sort_values("shortage", ascending=False)

            for _, r in receivers.iterrows():
                need = float(r["shortage"])
                if need < self.p.min_move:
                    continue
                best = None
                best_lane = None
                for _, d in donors.iterrows():
                    surplus = float(d["surplus"])
                    if surplus < self.p.min_move:
                        continue
                    lane_key = (d["wh"], r["wh"])
                    lane = lane_lookup.get(lane_key)
                    if lane is None:
                        continue
                    eff = float(lane["cost_per_unit"]) + self.p.cost_weight_leadtime * float(lane["lead_days"])
                    if best is None or eff < best:
                        best = eff
                        best_lane = (d, lane)
                if best_lane is None:
                    continue
                drow, lane = best_lane
                qty = min(float(drow["surplus"]), need)
                qty = int(np.floor(qty))
                if qty < self.p.min_move:
                    continue
                eta_day = day + int(lane["lead_days"])
                est_cost = qty * float(lane["cost_per_unit"])
                est_benefit = qty * (SKU_UNIT_VALUE[sku] * MARGIN_RATE)
                orders.append({
                    "order_day": day,
                    "order_date": date,
                    "from_wh": drow["wh"],
                    "to_wh": r["wh"],
                    "sku": sku,
                    "qty": qty,
                    "eta_day": eta_day,
                    "eta_date": None,
                    "reason": f"Projected shortage at {r['wh']} ({int(need)}u) and surplus at {drow['wh']} ({int(drow['surplus'])}u); lowest lane cost/leadtime.",
                    "est_benefit": float(est_benefit),
                    "est_cost": float(est_cost),
                })
                # Rather than updating DataFrame, just adjust in local variable
                donors.loc[donors["wh"] == drow["wh"], "surplus"] -= qty
                need -= qty

        return pd.DataFrame(orders)
