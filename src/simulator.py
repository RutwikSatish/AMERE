import pandas as pd
from .config import WAREHOUSES, SKUS, BASE_DEMAND, WH_MULT


def initial_inventory():
    rows = []
    for wh in WAREHOUSES:
        for sku in SKUS:
            init = round(12 * BASE_DEMAND[sku] * WH_MULT[wh])
            rows.append({"wh": wh, "sku": sku, "on_hand": int(init)})
    return pd.DataFrame(rows)


def run_simulation(days: int, demand_df: pd.DataFrame, lanes_df: pd.DataFrame, sku_df: pd.DataFrame, policy):
    inv = initial_inventory()
    transfers_all = []
    inv_states_all = []

    # in_transit records: eta_day, to_wh, sku, qty
    in_transit = []

    for day in range(1, days + 1):
        date = demand_df.loc[demand_df["day"] == day, "date"].iloc[0]

        # ---------- RECEIVE ARRIVALS ----------
        if len(in_transit) > 0:
            arrivals = pd.DataFrame(in_transit)
            arrivals_today = arrivals[arrivals["eta_day"] == day].copy()
            remaining = arrivals[arrivals["eta_day"] != day].copy()
            in_transit = remaining.to_dict("records")
        else:
            arrivals_today = pd.DataFrame(columns=["eta_day", "to_wh", "sku", "qty"])

        if not arrivals_today.empty:
            add = arrivals_today.groupby(["to_wh", "sku"], as_index=False)["qty"].sum()
            add = add.rename(columns={"to_wh": "wh"})
            inv = inv.merge(add, on=["wh", "sku"], how="left")
            inv["qty"] = inv["qty"].fillna(0).astype(int)
            inv["available"] = inv["on_hand"] + inv["qty"]
            inv = inv.drop(columns=["qty"])
        else:
            inv["available"] = inv["on_hand"]

        # ---------- DEMAND + FULFILLMENT (LOST SALES) ----------
        today_dem = demand_df.loc[demand_df["day"] == day, ["wh", "sku", "demand"]]
        inv = inv.merge(today_dem, on=["wh", "sku"], how="left")
        inv["demand"] = inv["demand"].fillna(0).astype(int)

        inv["fulfilled"] = inv[["available", "demand"]].min(axis=1)
        inv["lost_sales_units"] = (inv["demand"] - inv["available"]).clip(lower=0)
        inv["on_hand_after_sales"] = inv["available"] - inv["fulfilled"]

        # ---------- POLICY: PLAN TRANSFERS ----------
        on_hand_after = inv[["wh", "sku", "on_hand_after_sales"]].rename(columns={"on_hand_after_sales": "on_hand"})
        demand_hist = demand_df[demand_df["day"] <= day].copy()

        orders_df = policy.plan_transfers(
            day=day,
            date=date,
            on_hand_after_sales=on_hand_after,
            demand_hist=demand_hist,
            lanes=lanes_df,
        )

        if orders_df is None or orders_df.empty:
            orders_df = pd.DataFrame(columns=["order_day", "order_date", "from_wh", "to_wh", "sku", "qty", "eta_day"])

        # ---------- SHIP TRANSFERS ----------
        if not orders_df.empty:
            ship = orders_df.groupby(["from_wh", "sku"], as_index=False)["qty"].sum()
            ship = ship.rename(columns={"qty": "qty_ship"})

            inv = inv.merge(
                ship,
                left_on=["wh", "sku"],
                right_on=["from_wh", "sku"],
                how="left",
            )

            inv["qty_ship"] = inv["qty_ship"].fillna(0).astype(int)
            inv["on_hand_end"] = (inv["on_hand_after_sales"] - inv["qty_ship"]).clip(lower=0)

            inv = inv.drop(columns=["from_wh", "qty_ship"], errors="ignore")

            # add to in_transit
            for _, o in orders_df.iterrows():
                in_transit.append(
                    {
                        "eta_day": int(o["eta_day"]),
                        "to_wh": o["to_wh"],
                        "sku": o["sku"],
                        "qty": int(o["qty"]),
                    }
                )

            transfers_all.append(orders_df)
        else:
            inv["on_hand_end"] = inv["on_hand_after_sales"]

        # ---------- RECORD STATE ----------
        inv_states_all.append(
            inv.assign(day=day, date=date)[
                ["date", "day", "wh", "sku", "on_hand", "available", "demand", "fulfilled", "lost_sales_units", "on_hand_end"]
            ]
        )

        # ---------- ADVANCE INVENTORY ----------
        inv = inv[["wh", "sku", "on_hand_end"]].rename(columns={"on_hand_end": "on_hand"})

    transfers_df = pd.concat(transfers_all, ignore_index=True) if transfers_all else pd.DataFrame()
    inv_states_df = pd.concat(inv_states_all, ignore_index=True)

    return inv_states_df, transfers_df
