"""Chronological Random Forest and optional XGBoost training/evaluation."""
import argparse,json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,precision_recall_fscore_support
from ml.preprocessing import load_inventory
from ml.feature_engineering import add_features,add_overstock_target,FEATURES
ROOT=Path(__file__).resolve().parents[1]
def matrix(df,columns=None):
    raw=df[FEATURES+["product_id","store_id","category"]].copy()
    X=__import__("pandas").get_dummies(raw,columns=["product_id","store_id","category"],dtype=float)
    return X.reindex(columns=columns,fill_value=0) if columns is not None else X
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default=str(ROOT/'dataset'/'inventory.csv')); ap.add_argument('--model',choices=['random_forest','xgboost','both'],default='both'); ap.add_argument('--threshold-days',type=float,default=7.0); a=ap.parse_args()
    engineered=add_features(load_inventory(a.data))
    # Exclude the first observation per product/store, which has no trailing history.
    engineered=engineered[engineered.demand_7d_mean>0].copy()
    df=add_overstock_target(engineered,a.threshold_days)
    cutoff=df.date.quantile(.8); train=df.date<=cutoff; X=matrix(df); y=df.overstock
    Xtr,Xte,ytr,yte=X[train],X[~train],y[train],y[~train]
    if ytr.nunique()<2 or yte.nunique()<2: raise ValueError('Both classes must appear in train and holdout; check threshold/date coverage.')
    models={}
    if a.model in ('random_forest','both'): models['random_forest']=RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1)
    if a.model in ('xgboost','both'):
        try:
            from xgboost import XGBClassifier
            models['xgboost']=XGBClassifier(n_estimators=250,max_depth=5,learning_rate=.05,subsample=.9,colsample_bytree=.9,eval_metric='logloss',random_state=42,n_jobs=-1)
        except ImportError:
            if a.model=='xgboost': raise SystemExit('Install XGBoost: python -m pip install xgboost')
            print('XGBoost not installed; training Random Forest only.')
    out=ROOT/'models'; out.mkdir(exist_ok=True); scores={}
    for name,model in models.items():
        model.fit(Xtr,ytr); pred=model.predict(Xte); p,r,f,_=precision_recall_fscore_support(yte,pred,average='binary',zero_division=0)
        m={'threshold_days':a.threshold_days,'precision':float(p),'recall':float(r),'f1':float(f),'report':classification_report(yte,pred,zero_division=0,output_dict=True)}
        joblib.dump({'model':model,'columns':list(X.columns),'threshold_days':a.threshold_days},out/f'{name}_overstock.joblib')
        (out/f'{name}_metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8'); scores[name]=m
        print(f"{name}: precision={p:.3f} recall={r:.3f} F1={f:.3f}")
    if scores: print('Best by F1:',max(scores,key=lambda n:scores[n]['f1']))
if __name__=='__main__': main()
