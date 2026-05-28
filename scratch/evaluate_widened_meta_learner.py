import subprocess
import sys
import pathlib

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
    
joblib.dump(st_base_preds, '/tmp/base_test_preds.joblib')
print("--- Subprocess 1 completed successfully! ---", flush=True)
"""
    # Run python in a separate process to avoid OpenMP clashes with PyTorch
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("Error in LGBM prediction step:")
        print(proc.stderr)
        sys.exit(1)

def run_pytorch_step():
    code = """
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from models.meta_learner import FastQuantileRegressor

print("--- Subprocess 2: PyTorch training and evaluation ---", flush=True)
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

# Train widened meta-learner models: q = 0.05, 0.5, 0.95
meta_models = {}
for q in [0.05, 0.5, 0.95]:
    print(f"   Training q = {q}...", flush=True)
    qr = FastQuantileRegressor(quantile=q, max_iter=1500, lr=0.01)
    qr.fit(x_meta_full, y_meta)
    meta_models[q] = qr
    
# Load test base predictions from Subprocess 1
st_base_preds = joblib.load('/tmp/base_test_preds.joblib')
stations = df['station_id'].unique()

all_actuals_all = []
all_preds_all = []
all_actuals_dl = []
all_preds_dl = []
st_dl_coverages = {}

all_noon_q0075 = []
all_noon_q05 = []

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
    for q in [0.05, 0.5, 0.95]:
        m = meta_models[q]
        preds_dict[f"q_{q}"] = X_arr @ m.coef_ + m.intercept_
    preds = pd.DataFrame(preds_dict, index=x_meta_st_full.index)
    
    # Monotonicity post-sort
    vals_sorted = np.sort(preds[['q_0.05', 'q_0.5', 'q_0.95']].values, axis=1)
    preds['q_0.05'], preds['q_0.5'], preds['q_0.95'] = vals_sorted[:, 0], vals_sorted[:, 1], vals_sorted[:, 2]
    
    # All hours
    st_actual_all = y_norm.values
    st_pred_all = preds[['q_0.05', 'q_0.5', 'q_0.95']].values
    all_actuals_all.append(st_actual_all)
    all_preds_all.append(st_pred_all)
    
    # Daylight
    dl = cos_zenith > 0.087
    pos = y_norm > 0
    test_filter = dl & pos
    
    st_actual_dl = y_norm[test_filter].values
    st_pred_dl = preds.loc[test_filter, ['q_0.05', 'q_0.5', 'q_0.95']].values
    
    if len(st_actual_dl) > 0:
        st_lo = st_pred_dl[:, 0]
        st_hi = st_pred_dl[:, 2]
        st_cov = ((st_actual_dl >= st_lo) & (st_actual_dl <= st_hi)).mean()
        st_dl_coverages[st_id] = st_cov
        all_actuals_dl.append(st_actual_dl)
        all_preds_dl.append(st_pred_dl)
        
    # Noon
    noon = cos_zenith > 0.9
    if noon.sum() > 0:
        all_noon_q0075.extend(preds.loc[noon, 'q_0.05'].values)
        all_noon_q05.extend(preds.loc[noon, 'q_0.5'].values)
        
# Combine
actual_all = np.concatenate(all_actuals_all)
preds_all = np.concatenate(all_preds_all)
actual_dl = np.concatenate(all_actuals_dl)
preds_dl = np.concatenate(all_preds_dl)

# Metrics
cov_all = ((actual_all >= preds_all[:, 0]) & (actual_all <= preds_all[:, 2])).mean()
cov_dl = ((actual_dl >= preds_dl[:, 0]) & (actual_dl <= preds_dl[:, 2])).mean()

def compute_pinball(y_true, y_pred, q):
    r = y_true - y_pred
    return float(np.mean(np.where(r >= 0, q * r, (q - 1.0) * r)))
    
pb_lo = compute_pinball(actual_all, preds_all[:, 0], 0.05)
pb_mid = compute_pinball(actual_all, preds_all[:, 1], 0.50)
pb_hi = compute_pinball(actual_all, preds_all[:, 2], 0.95)
mean_pb = (pb_lo + pb_mid + pb_hi) / 3.0

all_noon_q0075 = np.array(all_noon_q0075)
all_noon_q05 = np.array(all_noon_q05)
non_zero = all_noon_q05 > 1e-4
ratio_noon = np.mean(all_noon_q0075[non_zero] / all_noon_q05[non_zero]) if non_zero.sum() > 0 else 0.0

print("\\n" + "="*50)
print("RAPOR")
print("="*50)
print(f"1. Üretim-saati coverage (cos_zenith>0.087 ∧ y_norm>0): {cov_dl:.6f}")
print(f"2. Gece-dahil coverage: {cov_all:.6f}")
print(f"3. Pinball mean: {mean_pb:.6f}")
print(f"4. q01/q05 öğle (collapse): {ratio_noon:.6f}")
print("\\n5. İstasyon bazlı üretim-saati coverage tablosu:")
print(f"{'İstasyon':<25} | {'Üretim-saati Coverage':<25}")
print("-" * 55)
for st_id in sorted(st_dl_coverages.keys()):
    print(f"{st_id:<25} | {st_dl_coverages[st_id] * 100:.2f}%")
print("="*50)

# Gündüz coverage >= 78% ise kaydet
if cov_dl >= 0.78:
    print("\\nDaylight coverage >= 78%! Saving the widened meta modeller...", flush=True)
    payload = {
        "models": meta_models,
        "quantiles": [0.05, 0.5, 0.95],
        "columns": list(x_meta_full.columns),
        "oof_metrics": {
            "pinball_lo": pb_lo,
            "pinball_mid": pb_mid,
            "pinball_hi": pb_hi,
            "picp_all": cov_all,
            "picp_daylight": cov_dl
        }
    }
    joblib.dump(payload, 'data/processed/meta_models_v2.joblib', compress=3)
    print("meta_models_v2.joblib successfully updated!", flush=True)
else:
    print("\\nDaylight coverage is still < 78%. Widen further or review.", flush=True)
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("Error in PyTorch step:")
        print(proc.stderr)
        sys.exit(1)

def main():
    run_lgbm_step()
    run_pytorch_step()

if __name__ == '__main__':
    main()
