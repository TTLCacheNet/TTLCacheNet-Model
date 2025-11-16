from __future__ import annotations
import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.models import Model

SEED = 42
np.random.seed(SEED)

# ---------- 데이터 로드 ----------
CSV_PATH = "data/syntheticDataset_O50.csv"   # 필요시 경로 수정
print("[train] CWD:", os.getcwd(), flush=True)
print("[train] CSV exists:", os.path.exists(CSV_PATH), flush=True)
df = pd.read_csv(CSV_PATH)

# 컬럼 정규화
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "objectId": "object_ID",
    "object_id": "object_ID",
    "objectid": "object_ID",
    "requestTime": "request_time",
})
if not {"object_ID", "request_time"}.issubset(df.columns):
    raise RuntimeError(f"required columns missing. have={list(df.columns)}")
df = df.sort_values("request_time").reset_index(drop=True)

# ---------- 파라미터(데이터 적으면 값 줄이세요) ----------
m, k = 10, 5
window_size = 100
step = 10

# (중요) object_ids는 "고유값"이어야 합니다.
object_ids = df["object_ID"].unique()
object_ids.sort()

# ---------- split ----------
train_cut = int(len(df) * 0.7)
val_cut   = int(len(df) * 0.85)
train_df = df.iloc[:train_cut].reset_index(drop=True)
val_df   = df.iloc[train_cut:val_cut].reset_index(drop=True)
test_df  = df.iloc[val_cut:].reset_index(drop=True)

# ---------- 시퀀스 생성 ----------
from app.feature_builder import build_simple_sequences  # 재사용

X_seq_train, y_pop_train, y_next_train = build_simple_sequences(
    train_df, object_ids, m, k, window_size, step, time_col="request_time"
)
X_seq_val,   y_pop_val,   y_next_val   = build_simple_sequences(
    val_df,   object_ids, m, k, window_size, step, time_col="request_time"
)

def ensure_3d_single_channel(X, m):
    import numpy as np
    if X.ndim == 2:                  # (N, m) → (N, m, 1)
        return X.reshape(-1, m, 1)
    if X.ndim == 3 and X.shape[-1] > 1:  # (N, m, C) → C=1만 사용
        return X[..., :1]
    return X

X_seq_train = ensure_3d_single_channel(X_seq_train, m)
X_seq_val   = ensure_3d_single_channel(X_seq_val, m)

print("[debug] X_seq_train shape:", X_seq_train.shape)
print("[debug] X_seq_val   shape:", X_seq_val.shape)

# 학습에 충분한 샘플이 없을 수 있으니 가드
if X_seq_train.size == 0:
    raise RuntimeError("Not enough data to build sequences. Consider lowering window_size/m/k/step.")

# ---------- 인코더(사전학습) ----------
def build_encoder(m, head_units: int = 32):
    enc_in = Input(shape=(m,1), name="seq_in")
    x = LSTM(128, return_sequences=True, name="enc_lstm_1")(enc_in)
    x = Dropout(0.2, name="enc_do_1")(x)
    x = LSTM(64, name="enc_lstm_2")(x)
    emb = Dropout(0.2, name="emb_drop")(x)
    h = Dense(head_units, activation="relu", name="pre_head")(emb)
    next_hat = Dense(1, activation="softplus", name="next_hat")(h)
    pretrain_model = Model(enc_in, next_hat, name="encoder_pretrain")
    pretrain_model.compile(optimizer="adam", loss="mae")
    encoder = Model(enc_in, emb, name="encoder_only")
    return encoder, pretrain_model

def extract_embeddings(encoder, X_seq):
    return encoder.predict(X_seq, verbose=0)  # (n, 64)

# y_next를 (N*d,1)로 평탄화 후 NaN 제거
y_next_train_flat = y_next_train.reshape(-1,1)
y_next_val_flat   = y_next_val.reshape(-1,1)
mask_tr = np.isfinite(y_next_train_flat[:,0])
mask_va = np.isfinite(y_next_val_flat[:,0])
X_seq_train_fit = X_seq_train[mask_tr]
X_seq_val_fit   = X_seq_val[mask_va]
y_tr_log = np.log1p(y_next_train_flat[mask_tr])
y_va_log = np.log1p(y_next_val_flat[mask_va])

print("[train] pretrain samples:", len(X_seq_train_fit), "(val:", len(X_seq_val_fit), ")", flush=True)
encoder, pretrain_model = build_encoder(m)
batch_size = max(32, int(len(X_seq_train_fit) * 0.1))

# 빠른 확인을 위해 epochs를 낮춰 시작해보세요(예: 3)
pretrain_model.fit(
    X_seq_train_fit, y_tr_log,
    validation_data=(X_seq_val_fit, y_va_log),
    epochs=3, batch_size=batch_size, verbose=1
)

# ---------- 임베딩 추출 ----------
emb_train = extract_embeddings(encoder, X_seq_train)
emb_val   = extract_embeddings(encoder, X_seq_val)

# ---------- 다음 간격 회귀(GBDT) ----------
y_train_flat = y_next_train.reshape(-1,)
y_val_flat   = y_next_val.reshape(-1,)
mask_tr2 = np.isfinite(y_train_flat)
mask_va2 = np.isfinite(y_val_flat)

emb_train_next = emb_train[mask_tr2]
emb_val_next   = emb_val[mask_va2]
y_tr2 = np.log1p(y_train_flat[mask_tr2])
y_va2 = np.log1p(y_val_flat[mask_va2])

reg = lgb.LGBMRegressor(objective="mae", learning_rate=0.05, num_leaves=64,
                        n_estimators=500, random_state=SEED)
reg.fit(
    emb_train_next, y_tr2,
    eval_set=[(emb_val_next, y_va2)],
    eval_metric="l1",
    callbacks=[early_stopping(100), log_evaluation(100)]
)

y_pred_val = np.expm1(reg.predict(emb_val_next, num_iteration=reg.best_iteration_))
print("Next MAE:", mean_absolute_error(np.expm1(y_va2), y_pred_val), flush=True)

# ---------- 아티팩트 저장 ----------
Path("artifacts").mkdir(exist_ok=True)
encoder.save("artifacts/encoder.keras")
joblib.dump(reg, "artifacts/reg_next.pkl")
np.save("artifacts/object_ids.npy", object_ids)
meta = {"m": m, "k": k, "window_size": window_size, "step": step,
        "modelVersion": "ttl-lstm-gbdt-2025-11-15"}
with open("artifacts/meta.json","w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("Saved artifacts to ./artifacts", flush=True)
