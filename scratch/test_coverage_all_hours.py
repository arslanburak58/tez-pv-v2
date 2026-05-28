import joblib
import pandas as pd
import numpy as np

print("1. Loading base models...", flush=True)
bm = joblib.load('data/processed/base_models_v2.joblib')
print("2. Loading meta models...", flush=True)
mm = joblib.load('data/processed/meta_models_v2.joblib')
print("3. Loading parquet...", flush=True)
df = pd.read_parquet('data/processed/features_v2.parquet')
print("4. Loading splits...", flush=True)
splits = joblib.load('data/processed/splits.joblib')

test_indices = set(splits['test_indices'])
feature_cols = bm['lgbm_q01'].feature_names_in_.tolist()
flag_cols = ['GHI_is_missing', 'T_amb_is_missing', 'RH_is_missing']
stations = df['station_id'].unique()

all_actuals = []
all_preds = []

print("5. Starting loop...", flush=True)
print(f"{'İstasyon':<25} | {'Test Satırı':<12} | {'Tüm Saatler Cov':<15} | {'Gündüz/Üretim Cov':<18}", flush=True)
print("-" * 75, flush=True)

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
            
    x_meta = pd.DataFrame(meta_cols, index=st_test.index)
    x_meta_full = pd.concat([x_meta, st_test[flag_cols]], axis=1)
    
    X_arr = x_meta_full.astype(float).values
    preds_dict = {}
    for q in [0.1, 0.5, 0.9]:
        m = mm['models'][q]
        preds_dict[f"q_{q}"] = X_arr @ m.coef_ + m.intercept_
    preds = pd.DataFrame(preds_dict, index=st_test.index)
    
    vals_sorted = np.sort(preds[['q_0.1', 'q_0.5', 'q_0.9']].values, axis=1)
    preds['q_0.1'], preds['q_0.5'], preds['q_0.9'] = vals_sorted[:, 0], vals_sorted[:, 1], vals_sorted[:, 2]
    
    # Tüm saatler (gece dahil)
    st_actual_all = st_test['y_norm'].values
    st_pred_all = preds.values
    st_cov_all = ((st_actual_all >= st_pred_all[:, 0]) & (st_actual_all <= st_pred_all[:, 2])).mean()
    
    # Gündüz / üretim saatleri
    dl = st_test['cos_zenith'] > 0.087
    pos = st_test['y_norm'] > 0
    test_filter = dl & pos
    
    st_actual_dl = st_test.loc[test_filter, 'y_norm'].values
    st_pred_dl = preds.loc[test_filter].values
    st_cov_dl = ((st_actual_dl >= st_pred_dl[:, 0]) & (st_actual_dl <= st_pred_dl[:, 2])).mean() if len(st_actual_dl) > 0 else 0
    
    print(f"{st_id:<25} | {len(st_test):<12} | {st_cov_all * 100:.2f}% | {st_cov_dl * 100:.2f}%", flush=True)
    
    all_actuals.append(st_actual_all)
    all_preds.append(st_pred_all)

print("-" * 75, flush=True)
actual_arr = np.concatenate(all_actuals)
preds_arr = np.concatenate(all_preds)

preds_df = pd.DataFrame(preds_arr, columns=['q_0.1', 'q_0.5', 'q_0.9'])
actual_ser = pd.Series(actual_arr)

lo = preds_df['q_0.1'].values
hi = preds_df['q_0.9'].values
y = actual_ser.values
total_coverage = ((y >= lo) & (y <= hi)).mean()

print(f"Toplam Test Seti Kapsama Oranı (TÜM SAATLER - GECE DAHİL) (Ağırlıksız): {total_coverage * 100:.2f}%")
