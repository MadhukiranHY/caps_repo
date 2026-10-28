"""Lagged features and temporary overstock proxy target."""
import numpy as np
FEATURES=["stock_on_hand","demand_7d_mean","demand_30d_mean","day_of_week","month"]
def add_features(df):
    out=df.sort_values(["product_id","store_id","date"]).copy()
    g=out.groupby(["product_id","store_id"],sort=False)["demand"]
    out["demand_7d_mean"]=g.transform(lambda s:s.shift(1).rolling(7,min_periods=1).mean())
    out["demand_30d_mean"]=g.transform(lambda s:s.shift(1).rolling(30,min_periods=1).mean())
    out["day_of_week"]=out.date.dt.dayofweek; out["month"]=out.date.dt.month
    out[FEATURES]=out[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)
    return out
def add_overstock_target(df,cover_threshold=7.0):
    """Demo label: more than threshold days of cover using previous 7-day demand.
    Replace/confirm this assumption with the team's agreed business threshold.
    """
    out=df.copy(); out["days_of_inventory_proxy"]=out.stock_on_hand/out.demand_7d_mean.clip(lower=.1)
    out["overstock"]=(out.days_of_inventory_proxy>cover_threshold).astype(int)
    return out
