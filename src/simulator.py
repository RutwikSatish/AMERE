import pandas as pd
from .config import WAREHOUSES, SKUS, BASE_DEMAND, WH_MULT
from .config import PolicyParams

def initial_inventory():
    rows = []
    for wh in WAREHOUSES:
        for sku in SKUS:
            init = round(12 * BASE_DEMAND[sku] * WH_MULT[wh])
            rows.append({"wh": wh, "sku": sku, "on_hand": int(init)})
    return pd.DataFrame(rows)

def run_simulation(days: int, demand_df: pd.DataFrame, lanes_df: pd.DataFrame, sku_df: pd.DataFrame, policy):
    inv = initial_inventory()
    transfers = []  # all transfer orders
    inv_states = []

    # in-transit list: each row has eta_day, to_wh, sku, qty
    in_transit = []

    for day in range(1, days+1):
        date = demand_df[demand_df["day"] == day]["date"].iloc[0]

        # receive arrivals
        arrivals = pd.DataFrame(in_transit)
        if arrivals.empty:
            arrivals_today = pd.DataFrame(columns=["to_wh","sku","qty"])
        else:
            arrivals_today = arrivals[arrivals["eta_day"] == day].copy()
            in_transit = arrivals[arrivals["eta_day"] != day].to_dict("records")

        if not arrivals_today.empty:
            add = arrivals_today.groupby(["to_wh","sku"], as_index=False)["qty"].sum()
            inv = inv.merge(add, left_on=["wh","sku"], right_on=["to_wh","sku"], how="left")
            inv["qty"] = inv["qty"].fillna(0).astype(int)
            inv["available"] = inv["on_hand"] + inv["qty"]
            inv = inv.drop(columns=["to_wh","qty"])
        else:
            inv["available"] = inv["on_hand"]

        # fulfill demand (lost sales)
        today_dem = demand_df[demand_df["day"] == day][["wh","sku","demand"]]
        inv = inv.merge(today_dem, on=["wh","sku"], how="left")
        inv["demand"] = inv["demand"].fillna(0).astype(int)

        inv["fulfilled"] = inv[["available","demand"]].min(axis=1)
        inv["lost_sales_units"] = (inv["demand"] - inv["available"]).clip(lower=0)
        inv["on_hand_after_sales"] = inv["available"] - inv["fulfilled"]

        # policy plans transfers
        on_hand_after = inv[["wh","sku","on_hand_after_sales"]].rename(columns={"on_hand_after_sales":"on_hand"})
        demand_hist = demand_df[demand_df["day"] <= day].copy()

        orders_df = policy.plan_transfers(day=day, date=date, on_hand_after_sales=on_hand_after, demand_hist=demand_hist, lanes=lanes_df)
        if orders_df is None or orders_df.empty:
            orders_df = pd.DataFrame(columns=["from_wh","to_wh","sku","qty","eta_day"])

        # ship transfers: subtract from donors + add to in_transit
        if not orders_df.empty:
            # subtract shipped
            ship = orders_df.groupby(["from_wh","sku"], as_index=False)["qty"].sum()
            inv = inv.merge(ship, left_on=["wh","sku"], right_on=["from_wh","sku"], how="left")
            inv["qty_y"] = inv["qty_y"].fillna(0).astype(int)
            inv["on_hand_end"] = (inv["on_hand_after_sales"] - inv["qty_y"]).clip(lower=0)
            inv = inv.drop(columns=["from_wh","qty_y"])
            # add in-transit
            for _, o in orders_df.iterrows():
                o = o.to_dict()
                o["eta_date"] = None
                in_transit.append({"eta_day": int(o["eta_day"]), "to_wh": o["to_wh"], "sku": o["sku"], "qty": int(o["qty"])})
            transfers.append(orders_df)
        else:
            inv["on_hand_end"] = inv["on_hand_after_sales"]

        # record state
        inv_states.append(inv.assign(day=day, date=date)[[
            "date","day","wh","sku","on_hand","available","demand","fulfilled","lost_sales_units","on_hand_end"
        ]])

        # advance inventory
        inv = inv[["wh","sku","on_hand_end"]].rename(columns={"on_hand_end":"on_hand"})

    transfers_df = pd.concat(transfers, ignore_index=True) if transfers else pd.DataFrame()
    inv_states_df = pd.concat(inv_states, ignore_index=True)
    return inv_states_df, transfers_df
