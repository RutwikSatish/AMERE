from dataclasses import dataclass

WAREHOUSES = ["W1_West", "W2_Central", "W3_South", "W4_East"]
SKUS = ["S1","S2","S3","S4","S5","S6","S7","S8"]

SKU_UNIT_VALUE = {"S1":15,"S2":25,"S3":35,"S4":50,"S5":75,"S6":120,"S7":180,"S8":250}
MARGIN_RATE = 0.30  # lost sales penalty = unit_value * margin_rate

BASE_DEMAND = {"S1":18,"S2":14,"S3":10,"S4":8,"S5":6,"S6":4,"S7":3,"S8":2}
WH_MULT = {"W1_West":1.1,"W2_Central":1.0,"W3_South":0.9,"W4_East":1.2}

@dataclass
class PolicyParams:
    horizon_days: int = 7
    buffer_days: int = 2
    min_move: int = 5
    cost_weight_leadtime: float = 0.15  # adds leadtime penalty into donor selection
