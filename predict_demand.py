"""Forecast next-day demand for one product and store."""
import argparse
from pathlib import Path
import joblib,numpy as np
from ml.preprocessing import load_inventory
ROOT=Path(__file__).resolve().parents[1]
def predict_demand(demand_history,model_path=None,scaler_path=None,lookback=14):
    try: import tensorflow as tf
    except ImportError as e: raise RuntimeError('TensorFlow is required') from e
    model=tf.keras.models.load_model(model_path or ROOT/'models'/'demand_lstm.keras'); scaler=joblib.load(scaler_path or ROOT/'models'/'demand_scaler.joblib')
    v=np.asarray(demand_history,dtype='float32')[-lookback:].reshape(-1,1)
    if len(v)<lookback: raise ValueError(f'Provide at least {lookback} daily demand values')
    z=scaler.transform(v).reshape(1,lookback,1); y=model.predict(z,verbose=0)[0,0]
    return max(0.0,float(scaler.inverse_transform([[y]])[0,0]))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--product',required=True); ap.add_argument('--store',required=True); ap.add_argument('--data',default=str(ROOT/'dataset'/'inventory.csv')); ap.add_argument('--lookback',type=int,default=14); a=ap.parse_args()
    d=load_inventory(a.data); g=d[(d.product_id.astype(str)==a.product)&(d.store_id.astype(str)==a.store)].sort_values('date')
    if g.empty: raise SystemExit(f'No observations for {a.product} in {a.store}')
    y=predict_demand(g.demand.tail(a.lookback).tolist(),lookback=a.lookback); print(f'Forecast next-day demand for {a.product} at {a.store}: {y:.2f}')
if __name__=='__main__': main()
