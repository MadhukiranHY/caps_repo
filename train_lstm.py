"""Train LSTM demand model and report holdout MAE/RMSE."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from ml.preprocessing import load_inventory
from dl.preprocessing import prepare_series
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default=str(ROOT/'dataset'/'inventory.csv')); ap.add_argument('--epochs',type=int,default=15); ap.add_argument('--lookback',type=int,default=14); ap.add_argument('--stride',type=int,default=3); a=ap.parse_args()
    try: import tensorflow as tf
    except ImportError as e: raise SystemExit('TensorFlow required; install requirements.txt') from e
    X,y,Xv,yv,scaler=prepare_series(load_inventory(a.data),a.lookback,a.stride)
    model=tf.keras.Sequential([tf.keras.layers.Input(shape=(a.lookback,1)),tf.keras.layers.LSTM(32),tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam',loss='mse',metrics=['mae']); model.fit(X,y,validation_data=(Xv,yv),epochs=a.epochs,batch_size=128,callbacks=[tf.keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True)],verbose=1)
    pred=model.predict(Xv,verbose=0); truth=scaler.inverse_transform(yv.reshape(-1,1)).ravel(); forecast=scaler.inverse_transform(pred.reshape(-1,1)).ravel()
    metrics={'mae':float(mean_absolute_error(truth,forecast)),'rmse':float(np.sqrt(mean_squared_error(truth,forecast))),'lookback':a.lookback,'stride':a.stride}
    out=ROOT/'models'; out.mkdir(exist_ok=True); model.save(out/'demand_lstm.keras'); joblib.dump(scaler,out/'demand_scaler.joblib'); (out/'demand_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(f"MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}; saved model/scaler in {out}")
if __name__=='__main__': main()
