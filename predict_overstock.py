"""Predict risk for the latest inventory row of each product/store."""
import argparse
from pathlib import Path
import joblib
from ml.preprocessing import load_inventory
from ml.feature_engineering import add_features
from ml.train_overstock import matrix
ROOT=Path(__file__).resolve().parents[1]
def predict_overstock(data,model_path=None):
    bundle=joblib.load(model_path or ROOT/'models'/'random_forest_overstock.joblib'); df=add_features(data)
    mask=df.date.eq(df.groupby(['product_id','store_id']).date.transform('max')); result=df.loc[mask].copy()
    X=matrix(result,bundle['columns']); result['overstock_prediction']=bundle['model'].predict(X)
    if hasattr(bundle['model'],'predict_proba'): result['overstock_probability']=bundle['model'].predict_proba(X)[:,1]
    result['days_of_inventory_proxy']=result.stock_on_hand/result.demand_7d_mean.clip(lower=.1); result['threshold_days']=bundle['threshold_days']
    return result
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default=str(ROOT/'dataset'/'inventory.csv')); ap.add_argument('--model'); ap.add_argument('--output',default=str(ROOT/'models'/'overstock_predictions.csv')); a=ap.parse_args()
    result=predict_overstock(load_inventory(a.data),a.model); result.to_csv(a.output,index=False); print(f'Saved {len(result)} latest product/store predictions to {a.output}')
if __name__=='__main__': main()
