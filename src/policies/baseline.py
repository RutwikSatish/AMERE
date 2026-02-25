import pandas as pd

class BaselinePolicy:
    name = "Baseline (no transfers)"
    def plan_transfers(self, **kwargs) -> pd.DataFrame:
        # return empty transfer orders
        return pd.DataFrame(columns=["order_day","order_date","from_wh","to_wh","sku","qty","eta_day","eta_date","reason","est_benefit","est_cost"])
