def explain_transfers(transfers_df, inv_snapshot, sku_values, margin_rate):
    explanations = []

    for _, row in transfers_df.iterrows():
        sku = row["sku"]
        qty = row["qty"]
        from_wh = row["from_wh"]
        to_wh = row["to_wh"]

        penalty_per_unit = sku_values[sku] * margin_rate
        est_benefit = qty * penalty_per_unit
        est_cost = row.get("est_cost", 0)

        explanation = (
            f"{qty} units moved from {from_wh} to {to_wh}. "
            f"Expected lost sales avoided ≈ ${est_benefit:,.0f}. "
            f"Estimated transfer cost ≈ ${est_cost:,.0f}."
        )

        explanations.append(explanation)

    transfers_df["explanation"] = explanations
    return transfers_df
