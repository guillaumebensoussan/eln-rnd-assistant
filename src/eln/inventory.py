"""Module inventaire — déductions de stock et alertes (Pandas)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
INVENTORY_CSV = DATA_DIR / "inventory.csv"
LOW_STOCK_THRESHOLD_G = 20.0


class Inventory:
    def __init__(self, csv: Path | None = None):
        self.csv = csv or INVENTORY_CSV
        self.df = pd.read_csv(self.csv)

    def deduct(self, reagent_id: str, grams: float) -> dict:
        mask = self.df["reagent_id"] == reagent_id
        if not mask.any():
            return {"ok": False, "error": f"reagent {reagent_id} not found"}
        new_qty = float(self.df.loc[mask, "quantity_g"].iloc[0]) - grams
        new_qty = max(new_qty, 0.0)
        self.df.loc[mask, "quantity_g"] = round(new_qty, 1)
        return {
            "ok": True,
            "reagent_id": reagent_id,
            "remaining_g": new_qty,
            "low_stock": new_qty < LOW_STOCK_THRESHOLD_G,
        }

    def alerts(self) -> dict:
        low = self.df[self.df["quantity_g"] < LOW_STOCK_THRESHOLD_G]["reagent_id"].tolist()
        expired = self.df[pd.to_datetime(self.df["expiry"]).dt.date < date(2026, 5, 29)]["reagent_id"].tolist()
        return {"low_stock": low, "expired": expired}

    def save(self) -> None:
        self.df.to_csv(self.csv, index=False)


if __name__ == "__main__":
    inv = Inventory()
    print("deduct:", inv.deduct(inv.df["reagent_id"].iloc[0], 5))
    print("alerts:", {k: len(v) for k, v in inv.alerts().items()})
