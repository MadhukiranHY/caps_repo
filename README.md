# AI Inventory Project — supplied dataset edition

This version uses the inventory exports supplied in `sap.zip`: 76,000 dated product/store rows (20 products × 5 stores × 760 dates) and a 100-row current inventory handoff.

## Data provenance and overstock assumption

The supplied ZIP identifies these as processed data exports and says the original raw `sales_data.csv` is missing. `dataset/inventory.csv` is normalized from `05_inventory_demand_features.csv` with categories from `01_product.csv`; `inventory_current.csv` and `ml_ready_overstock.csv` are also retained. Do not represent these processed exports as the missing raw source.

No business overstock label/cutoff was supplied. For demonstration, the model uses **more than 7 days of cover**, calculated as stock divided by the preceding 7-day average demand. This is a temporary assumption; confirm or change it with your team lead before reporting it as a business rule. You can adjust it with `--threshold-days`.

## Windows setup

From this project folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use 64-bit Python 3.10–3.13. TensorFlow runs on CPU on native Windows.

## ML: overstock risk

```powershell
python -m ml.train_overstock
python -m ml.predict_overstock
```

Training compares Random Forest and XGBoost when installed, uses a chronological date split, reports precision/recall/F1, and saves models plus metric JSON files in `models`. Prediction writes the latest one row per product/store to `models\overstock_predictions.csv`.

## DL: demand forecast

```powershell
python -m dl.train_lstm --epochs 15 --lookback 14
python -m dl.predict_demand --product P0001 --store S001
```

The LSTM learns from each product/store demand sequence, holds out later dates, reports MAE/RMSE in units, and saves the model, scaler, and metrics in `models`. Stride 3 reduces CPU training time.

These are separate models sharing the same inventory history. Predictions demonstrate the workflow and need validation against the team's agreed definitions before operational use. The source notes that real reorder point and maximum stock values were not supplied.
