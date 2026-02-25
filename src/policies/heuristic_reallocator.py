import pandas as pd
import numpy as np
from ..config import PolicyParams, SKU_UNIT_VALUE, MARGIN_RATE

class HeuristicReallocatorPolicy:
    name = "Heuristic Reallocator"

    def __init__(self, params: PolicyParams):
        self.p = params

    def plan_transfers(self, day: int, date, on_hand_after_sales: pd.DataFrame,
                       demand_hist: pd.DataFrame, lanes: pd.DataFrame) -> pd.DataFrame:
        """
        on_hand_after_sales: columns [wh, sku, on_hand]
        demand_hist: demand rows up to current day (inclusive), columns [day, wh, sku, demand, shock_flag]
        lanes: [from_wh,to_wh,lead_days,cost_per_unit,cap_units]
        """
        orders = []

        # Forecast: avg last 7 days; if shock active recently, use last 3 days
        for sku in on_hand_after_sales["sku"].unique():
            inv = on_hand_after_sales[on_hand_after_sales["sku"] == sku].copy()

            # compute avg daily demand per wh
            recent7 = demand_hist[(demand_hist["sku"] == sku) & (demand_hist["day"] <= day)].copy()
            def avg_dem(wh):
                sub = recent7[recent7["wh"] == wh].tail(7)
                if len(sub) == 0:
                    return 0.0
                # if last 3 days had shock, adapt
                last3 = sub.tail(3)
                if last3["shock_flag"].sum() >= 1:
                    return float(last3["demand"].mean())
                return float(sub["demand"].mean())

            inv["avg_dem"] = inv["wh"].apply(avg_dem)
            inv["target"] = (self.p.horizon_days * inv["avg_dem"]) + (self.p.buffer_days * inv["avg_dem"])

            inv["surplus"] = (inv["on_hand"] - inv["target"]).clip(lower=0)
            inv["shortage"] = (inv["target"] - inv["on_hand"]).clip(lower=0)

            donors = inv[inv["surplus"] > 0].copy()
            recv = inv[inv["shortage"] > 0].copy()

            if donors.empty or recv.empty:
                continue

            recv = recv.sort_values("shortage", ascending=False)

            # allocate greedily
            for _, r in recv.iterrows():
                need = float(r["shortage"])
                if need < self.p.min_move:
                    continue

                # choose best donor by effective cost (lane cost + leadtime penalty)
                best = None
                best_lane = None
                for _, d in donors.iterrows():
                    if d["surplus"] < self.p.min_move:
                        continue
                    lane = lanes[(lanes["from_wh"] == d["wh"]) & (lanes["to_wh"] == r["wh"])].iloc[0]
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
                est_benefit = qty * (SKU_UNIT_VALUE[sku] * MARGIN_RATE)  # proxy benefit

                orders.append({
                    "order_day": day,
                    "order_date": date,
                    "from_wh": drow["wh"],
                    "to_wh": r["wh"],
                    "sku": sku,
                    "qty": qty,
                    "eta_day": eta_day,
                    "eta_date": None,  # filled later by simulator
                    "reason": f"Projected shortage at {r['wh']} ({int(need)}u) and surplus at {drow['wh']} ({int(drow['surplus'])}u); lowest lane cost/leadtime.",
                    "est_benefit": float(est_benefit),
                    "est_cost": float(est_cost),
                })

                # update local donor/receiver needs
                donors.loc[donors["wh"] == drow["wh"], "surplus"] -= qty
                need -= qty

        return pd.DataFrame(orders)
