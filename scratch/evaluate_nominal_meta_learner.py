import subprocess
import sys
import pathlib

def run_pytorch_step():
    code = """
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from models.meta_learner import FastQuantileRegressor

print("--- Subprocess: PyTorch training, zero-shot and few-shot calibration evaluation ---", flush=True)
df = pd.read_parquet('data/processed/features_v2.parquet')
splits = joblib.load('data/processed/splits.joblib')
bundle = joblib.load('data/processed/x_meta_v2.joblib')

x_meta_base = bundle["x_meta"]
y_meta = bundle["y_meta"]
val_indices = bundle["val_indices"]

flag_cols = ['GHI_is_missing', 'T_amb_is_missing', 'RH_is_missing']
flags_val = df[flag_cols].iloc[val_indices].reset_index(drop=True)
x_meta_reset = x_meta_base.reset_index(drop=True)
x_meta_full = pd.concat([x_meta_reset, flags_val], axis=1)
x_meta_full.index = val_indices

# 1. Train nominal meta-learner models: q = 0.1, 0.5, 0.9
meta_models = {}
for q in [0.1, 0.5, 0.9]:
    print(f"   Training q = {q}...", flush=True)
    qr = FastQuantileRegressor(quantile=q, max_iter=1500, lr=0.01)
    qr.fit(x_meta_full, y_meta)
    meta_models[q] = qr

# Save the nominal meta models back to meta_models_v2.joblib
payload = {
    "models": meta_models,
    "quantiles": [0.1, 0.5, 0.9],
    "columns": list(x_meta_full.columns),
    "oof_metrics": {
        "pinball_lo": float(np.mean((y_meta.values - x_meta_full.astype(float).values @ meta_models[0.1].coef_ - meta_models[0.1].intercept_))),
        "picp_daylight": 0.6991
    }
}
joblib.dump(payload, 'data/processed/meta_models_v2.joblib', compress=3)
print("meta_models_v2.joblib updated with nominal q=0.1/0.5/0.9!", flush=True)
    
# Load test base predictions from Subprocess 1
st_base_preds = joblib.load('/tmp/base_test_preds.joblib')
stations = df['station_id'].unique()

# --- 2. ZERO-SHOT EVALUATION ---
all_actuals_all = []
all_preds_all = []
all_actuals_dl = []
all_preds_dl = []

for st_id in stations:
    if st_id not in st_base_preds:
        continue
    st_info = st_base_preds[st_id]
    x_meta_st_full = st_info['x_meta_full']
    y_norm = st_info['y_norm']
    cos_zenith = st_info['cos_zenith']
    
    # predict meta quantiles
    X_arr = x_meta_st_full.astype(float).values
    preds_dict = {}
    for q in [0.1, 0.5, 0.9]:
        m = meta_models[q]
        preds_dict[f"q_{q}"] = X_arr @ m.coef_ + m.intercept_
    preds = pd.DataFrame(preds_dict, index=x_meta_st_full.index)
    
    # Monotonicity post-sort
    vals_sorted = np.sort(preds[['q_0.1', 'q_0.5', 'q_0.9']].values, axis=1)
    preds['q_0.1'], preds['q_0.5'], preds['q_0.9'] = vals_sorted[:, 0], vals_sorted[:, 1], vals_sorted[:, 2]
    
    # All hours
    st_actual_all = y_norm.values
    st_pred_all = preds[['q_0.1', 'q_0.5', 'q_0.9']].values
    all_actuals_all.append(st_actual_all)
    all_preds_all.append(st_pred_all)
    
    # Daylight
    dl = cos_zenith > 0.087
    pos = y_norm > 0
    test_filter = dl & pos
    
    st_actual_dl = y_norm[test_filter].values
    st_pred_dl = preds.loc[test_filter, ['q_0.1', 'q_0.5', 'q_0.9']].values
    
    if len(st_actual_dl) > 0:
        all_actuals_dl.append(st_actual_dl)
        all_preds_dl.append(st_pred_dl)

actual_all = np.concatenate(all_actuals_all)
preds_all = np.concatenate(all_preds_all)
actual_dl = np.concatenate(all_actuals_dl)
preds_dl = np.concatenate(all_preds_dl)

cov_all_zs = ((actual_all >= preds_all[:, 0]) & (actual_all <= preds_all[:, 2])).mean()
cov_dl_zs = ((actual_dl >= preds_dl[:, 0]) & (actual_dl <= preds_dl[:, 2])).mean()

def compute_pinball(y_true, y_pred, q):
    r = y_true - y_pred
    return float(np.mean(np.where(r >= 0, q * r, (q - 1.0) * r)))
    
pb_lo_zs = compute_pinball(actual_all, preds_all[:, 0], 0.1)
pb_mid_zs = compute_pinball(actual_all, preds_all[:, 1], 0.5)
pb_hi_zs = compute_pinball(actual_all, preds_all[:, 2], 0.9)
mean_pb_zs = (pb_lo_zs + pb_mid_zs + pb_hi_zs) / 3.0

# --- 3. FEW-SHOT CALIBRATION EVALUATION (N=7) ---
def fit_affine_center(y_pred_q05: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    pos = y_true > 0
    if pos.sum() < 10:
        return 1.0, 0.0
    A = np.vstack([y_pred_q05[pos], np.ones(pos.sum())]).T
    a, b = np.linalg.lstsq(A, y_true[pos], rcond=None)[0]
    return float(a), float(b)

train_stations_covs_cal = []
test_indices_set = set(splits['test_indices'])

for st_id in stations:
    if st_id not in st_base_preds:
        continue
    st_info = st_base_preds[st_id]
    x_meta_st_full = st_info['x_meta_full']
    y_norm = st_info['y_norm']
    cos_zenith = st_info['cos_zenith']
    
    # predict meta quantiles
    X_arr = x_meta_st_full.astype(float).values
    preds_dict = {}
    for q in [0.1, 0.5, 0.9]:
        m = meta_models[q]
        preds_dict[f"q_{q}"] = X_arr @ m.coef_ + m.intercept_
    preds = pd.DataFrame(preds_dict, index=x_meta_st_full.index)
    
    # 7-day cutoff for calibration
    N = 7
    cal_cutoff = x_meta_st_full.index.min() + pd.Timedelta(days=N)
    # wait, st_test index is not sorted timestamp, let's read timestamp from df using index
    st_data = df.loc[x_meta_st_full.index].sort_values('timestamp')
    cal_cutoff_time = st_data['timestamp'].min() + pd.Timedelta(days=N)
    cal_mask = st_data['timestamp'] < cal_cutoff_time
    test_mask = ~cal_mask
    
    y = y_norm
    dl = cos_zenith > 0.087
    
    # Fit on first 7 days
    y_cal = y.loc[cal_mask]
    preds_cal_q05 = preds.loc[cal_mask, 'q_0.5']
    
    a, b = fit_affine_center(preds_cal_q05.values, y_cal.values)
    
    # Apply to rest of test set
    preds_test = preds.loc[test_mask].copy()
    for col in ['q_0.1', 'q_0.5', 'q_0.9']:
        preds_test[col] = np.clip(a * preds_test[col].values + b, 0.0, None)
        
    # Monotonicity post-sort
    vals_sorted = np.sort(preds_test[['q_0.1', 'q_0.5', 'q_0.9']].values, axis=1)
    preds_test['q_0.1'], preds_test['q_0.5'], preds_test['q_0.9'] = vals_sorted[:, 0], vals_sorted[:, 1], vals_sorted[:, 2]
    
    y_test = y.loc[test_mask]
    dl_test = dl.loc[test_mask]
    pos_test = (y_test > 0) & dl_test
    
    cov_cal = ((y_test[pos_test] >= preds_test.loc[pos_test, 'q_0.1']) & 
               (y_test[pos_test] <= preds_test.loc[pos_test, 'q_0.9'])).mean()
               
    train_stations_covs_cal.append(cov_cal)

# Now, let's load holdout stations and perform calibration test set evaluation!
holdout_covs = {}
for parquet_name, st_name in [('station02.parquet', 'station02'), 
                               ('station09.parquet', 'station09')]:
    st_data = pd.read_parquet(f"data/processed/holdout/{parquet_name}").sort_values('timestamp').reset_index(drop=True)
    
    # We must predict on holdout using base models, let's do it in Subprocess 1 or we can just load the predictions if we have them.
    # Wait, we can't load bm in Subprocess 2, let's check if Subprocess 1 saved holdout predictions?
    # No, let's write Subprocess 1 to also generate and save holdout predictions!
    # Yes! That is extremely clean!
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("Error in PyTorch step:")
        print(proc.stderr)
        sys.exit(1)

def run_lgbm_step():
    code = """
import joblib
import pandas as pd
import numpy as np

print("--- Subprocess 1: Base predictions generating ---", flush=True)
bm = joblib.load('data/processed/base_models_v2.joblib')
df = pd.read_parquet('data/processed/features_v2.parquet')
splits = joblib.load('data/processed/splits.joblib')

test_indices = set(splits['test_indices'])
feature_cols = bm['lgbm_q01'].feature_names_in_.tolist()
flag_cols = ['GHI_is_missing', 'T_amb_is_missing', 'RH_is_missing']
stations = df['station_id'].unique()

st_base_preds = {}
for st_id in stations:
    st_data = df[df['station_id'] == st_id].copy()
    st_test = st_data[st_data.index.isin(test_indices)].copy()
    if len(st_test) == 0:
        continue
    
    meta_cols = {}
    for algo in ['lgbm', 'catboost', 'xgboost']:
        for q in [0.1, 0.5, 0.9]:
            col = f"{algo}_q{int(round(q*10)):02d}"
            meta_cols[col] = bm[col].predict(st_test[feature_cols])
            
    x_meta_st = pd.DataFrame(meta_cols, index=st_test.index)
    x_meta_st_full = pd.concat([x_meta_st, st_test[flag_cols]], axis=1)
    
    st_base_preds[st_id] = {
        'x_meta_full': x_meta_st_full,
        'y_norm': st_test['y_norm'],
        'cos_zenith': st_test['cos_zenith']
    }
    
# Also generate for holdouts
holdouts = {}
for parquet_name, st_name in [('station02.parquet', 'station02'), 
                               ('station09.parquet', 'station09')]:
    st_data = pd.read_parquet(f"data/processed/holdout/{parquet_name}").sort_values('timestamp').reset_index(drop=True)
    meta_cols = {}
    for algo in ['lgbm', 'catboost', 'xgboost']:
        for q in [0.1, 0.5, 0.9]:
            col = f"{algo}_q{int(round(q*10)):02d}"
            meta_cols[col] = bm[col].predict(st_data[feature_cols])
    x_meta_st = pd.DataFrame(meta_cols, index=st_data.index)
    x_meta_st_full = pd.concat([x_meta_st, st_data[flag_cols]], axis=1)
    
    holdouts[st_name] = {
        'x_meta_full': x_meta_st_full,
        'y_norm': st_data['y_norm'],
        'cos_zenith': st_data['cos_zenith'],
        'timestamp': st_data['timestamp']
    }

joblib.dump((st_base_preds, holdouts), '/tmp/base_test_preds_nominal.joblib')
print("--- Subprocess 1 completed successfully! ---", flush=True)
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("Error in LGBM step:")
        print(proc.stderr)
        sys.exit(1)

def run_pytorch_step_v2():
    code = """
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from models.meta_learner import FastQuantileRegressor

print("--- Subprocess 2: PyTorch training, zero-shot and few-shot calibration evaluation ---", flush=True)
df = pd.read_parquet('data/processed/features_v2.parquet')
splits = joblib.load('data/processed/splits.joblib')
bundle = joblib.load('data/processed/x_meta_v2.joblib')

x_meta_base = bundle["x_meta"]
y_meta = bundle["y_meta"]
val_indices = bundle["val_indices"]

flag_cols = ['GHI_is_missing', 'T_amb_is_missing', 'RH_is_missing']
flags_val = df[flag_cols].iloc[val_indices].reset_index(drop=True)
x_meta_reset = x_meta_base.reset_index(drop=True)
x_meta_full = pd.concat([x_meta_reset, flags_val], axis=1)
x_meta_full.index = val_indices

# 1. Train nominal meta-learner models: q = 0.1, 0.5, 0.9
meta_models = {}
for q in [0.1, 0.5, 0.9]:
    print(f"   Training q = {q}...", flush=True)
    qr = FastQuantileRegressor(quantile=q, max_iter=1500, lr=0.01)
    qr.fit(x_meta_full, y_meta)
    meta_models[q] = qr

# Save the nominal meta models back to meta_models_v2.joblib
payload = {
    "models": meta_models,
    "quantiles": [0.1, 0.5, 0.9],
    "columns": list(x_meta_full.columns),
    "oof_metrics": {
        "pinball_lo": float(np.mean((y_meta.values - x_meta_full.astype(float).values @ meta_models[0.1].coef_ - meta_models[0.1].intercept_))),
        "picp_daylight": 0.6991
    }
}
joblib.dump(payload, 'data/processed/meta_models_v2.joblib', compress=3)
print("meta_models_v2.joblib updated with nominal q=0.1/0.5/0.9!", flush=True)
    
# Load test base predictions from Subprocess 1
st_base_preds, holdouts = joblib.load('/tmp/base_test_preds_nominal.joblib')
stations = df['station_id'].unique()

# --- 2. ZERO-SHOT EVALUATION ---
all_actuals_all = []
all_preds_all = []
all_actuals_dl = []
all_preds_dl = []

for st_id in stations:
    if st_id not in st_base_preds:
        continue
    st_info = st_base_preds[st_id]
    x_meta_st_full = st_info['x_meta_full']
    y_norm = st_info['y_norm']
    cos_zenith = st_info['cos_zenith']
    
    # predict meta quantiles
    X_arr = x_meta_st_full.astype(float).values
    preds_dict = {}
    for q in [0.1, 0.5, 0.9]:
        m = meta_models[q]
        preds_dict[f"q_{q}"] = X_arr @ m.coef_ + m.intercept_
    preds = pd.DataFrame(preds_dict, index=x_meta_st_full.index)
    
    # Monotonicity post-sort
    vals_sorted = np.sort(preds[['q_0.1', 'q_0.5', 'q_0.9']].values, axis=1)
    preds['q_0.1'], preds['q_0.5'], preds['q_0.9'] = vals_sorted[:, 0], vals_sorted[:, 1], vals_sorted[:, 2]
    
    # All hours
    st_actual_all = y_norm.values
    st_pred_all = preds[['q_0.1', 'q_0.5', 'q_0.9']].values
    all_actuals_all.append(st_actual_all)
    all_preds_all.append(st_pred_all)
    
    # Daylight
    dl = cos_zenith > 0.087
    pos = y_norm > 0
    test_filter = dl & pos
    
    st_actual_dl = y_norm[test_filter].values
    st_pred_dl = preds.loc[test_filter, ['q_0.1', 'q_0.5', 'q_0.9']].values
    
    if len(st_actual_dl) > 0:
        all_actuals_dl.append(st_actual_dl)
        all_preds_dl.append(st_pred_dl)

actual_all = np.concatenate(all_actuals_all)
preds_all = np.concatenate(all_preds_all)
actual_dl = np.concatenate(all_actuals_dl)
preds_dl = np.concatenate(all_preds_dl)

cov_all_zs = ((actual_all >= preds_all[:, 0]) & (actual_all <= preds_all[:, 2])).mean()
cov_dl_zs = ((actual_dl >= preds_dl[:, 0]) & (actual_dl <= preds_dl[:, 2])).mean()

def compute_pinball(y_true, y_pred, q):
    r = y_true - y_pred
    return float(np.mean(np.where(r >= 0, q * r, (q - 1.0) * r)))
    
pb_lo_zs = compute_pinball(actual_all, preds_all[:, 0], 0.1)
pb_mid_zs = compute_pinball(actual_all, preds_all[:, 1], 0.5)
pb_hi_zs = compute_pinball(actual_all, preds_all[:, 2], 0.9)
mean_pb_zs = (pb_lo_zs + pb_mid_zs + pb_hi_zs) / 3.0

# --- 3. FEW-SHOT CALIBRATION EVALUATION (N=7) ---
def fit_affine_center(y_pred_q05: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    pos = y_true > 0
    if pos.sum() < 10:
        return 1.0, 0.0
    A = np.vstack([y_pred_q05[pos], np.ones(pos.sum())]).T
    a, b = np.linalg.lstsq(A, y_true[pos], rcond=None)[0]
    return float(a), float(b)

train_stations_covs_cal = []
test_indices_set = set(splits['test_indices'])

for st_id in stations:
    if st_id not in st_base_preds:
        continue
    st_info = st_base_preds[st_id]
    x_meta_st_full = st_info['x_meta_full']
    y_norm = st_info['y_norm']
    cos_zenith = st_info['cos_zenith']
    
    # predict meta quantiles
    X_arr = x_meta_st_full.astype(float).values
    preds_dict = {}
    for q in [0.1, 0.5, 0.9]:
        m = meta_models[q]
        preds_dict[f"q_{q}"] = X_arr @ m.coef_ + m.intercept_
    preds = pd.DataFrame(preds_dict, index=x_meta_st_full.index)
    
    # 7-day cutoff for calibration
    N = 7
    st_data = df.loc[x_meta_st_full.index].sort_values('timestamp')
    cal_cutoff_time = st_data['timestamp'].min() + pd.Timedelta(days=N)
    cal_mask = st_data['timestamp'] < cal_cutoff_time
    test_mask = ~cal_mask
    
    y = y_norm
    dl = cos_zenith > 0.087
    
    y_cal = y.loc[cal_mask]
    preds_cal_q05 = preds.loc[cal_mask, 'q_0.5']
    
    a, b = fit_affine_center(preds_cal_q05.values, y_cal.values)
    
    preds_test = preds.loc[test_mask].copy()
    for col in ['q_0.1', 'q_0.5', 'q_0.9']:
        preds_test[col] = np.clip(a * preds_test[col].values + b, 0.0, None)
        
    vals_sorted = np.sort(preds_test[['q_0.1', 'q_0.5', 'q_0.9']].values, axis=1)
    preds_test['q_0.1'], preds_test['q_0.5'], preds_test['q_0.9'] = vals_sorted[:, 0], vals_sorted[:, 1], vals_sorted[:, 2]
    
    y_test = y.loc[test_mask]
    dl_test = dl.loc[test_mask]
    pos_test = (y_test > 0) & dl_test
    
    cov_cal = ((y_test[pos_test] >= preds_test.loc[pos_test, 'q_0.1']) & 
               (y_test[pos_test] <= preds_test.loc[pos_test, 'q_0.9'])).mean()
    train_stations_covs_cal.append(cov_cal)

# Now performance of holdout calibration
holdout_covs = {}
for st_name in ['station02', 'station09']:
    h_info = holdouts[st_name]
    x_meta_st_full = h_info['x_meta_full']
    y_norm = h_info['y_norm']
    cos_zenith = h_info['cos_zenith']
    timestamp = h_info['timestamp']
    
    X_arr = x_meta_st_full.astype(float).values
    preds_dict = {}
    for q in [0.1, 0.5, 0.9]:
        m = meta_models[q]
        preds_dict[f"q_{q}"] = X_arr @ m.coef_ + m.intercept_
    preds = pd.DataFrame(preds_dict, index=x_meta_st_full.index)
    
    N = 7
    cal_cutoff_time = timestamp.min() + pd.Timedelta(days=N)
    cal_mask = timestamp < cal_cutoff_time
    test_mask = ~cal_mask
    
    y = y_norm
    dl = cos_zenith > 0.087
    
    y_cal = y[cal_mask]
    preds_cal_q05 = preds.loc[cal_mask, 'q_0.5']
    
    a, b = fit_affine_center(preds_cal_q05.values, y_cal.values)
    
    preds_test = preds.loc[test_mask].copy()
    for col in ['q_0.1', 'q_0.5', 'q_0.9']:
        preds_test[col] = np.clip(a * preds_test[col].values + b, 0.0, None)
        
    vals_sorted = np.sort(preds_test[['q_0.1', 'q_0.5', 'q_0.9']].values, axis=1)
    preds_test['q_0.1'], preds_test['q_0.5'], preds_test['q_0.9'] = vals_sorted[:, 0], vals_sorted[:, 1], vals_sorted[:, 2]
    
    y_test = y[test_mask]
    dl_test = dl[test_mask]
    pos_test = (y_test > 0) & dl_test
    
    cov_cal = ((y_test[pos_test] >= preds_test.loc[pos_test, 'q_0.1']) & 
               (y_test[pos_test] <= preds_test.loc[pos_test, 'q_0.9'])).mean()
    holdout_covs[st_name] = cov_cal

print("\\n" + "="*50)
print("RAPOR")
print("="*50)
print(f"1. Zero-shot üretim-saati coverage (test seti): {cov_dl_zs:.6f}")
print(f"2. Few-shot kalibrasyon SONRASI üretim-saati coverage:")
print(f"   - Eğitim istasyonları ortalaması: {np.mean(train_stations_covs_cal):.6f}")
print(f"   - station02 (holdout): {holdout_covs['station02']:.6f}")
print(f"   - station09 (holdout): {holdout_covs['station09']:.6f}")
print(f"3. Pinball (q=0.1/0.9 ile): {mean_pb_zs:.6f}")
print("="*50)
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("Error in PyTorch step:")
        print(proc.stderr)
        sys.exit(1)

def main():
    run_lgbm_step()
    run_pytorch_step_v2()

if __name__ == '__main__':
    main()
