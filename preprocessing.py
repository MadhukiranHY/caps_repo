"""Load and validate product/store/day inventory history."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
REQUIRED={"date","product_id","store_id","stock_on_hand","demand"}
def load_inventory(path=None):
    path=Path(path) if path else ROOT/"dataset"/"inventory.csv"
    df=pd.read_csv(path); missing=REQUIRED-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    for c in ("stock_on_hand","demand"): df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["date","product_id","store_id","stock_on_hand","demand"]).copy()
    if (df[["stock_on_hand","demand"]]<0).any().any(): raise ValueError("Stock and demand must be non-negative")
    return df.sort_values(["product_id","store_id","date"]).reset_index(drop=True)
