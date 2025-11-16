from __future__ import annotations
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any
from tensorflow.keras.models import load_model

class Artifacts:
    def __init__(self, dirpath: str = "artifacts"):
        p = Path(dirpath)
        self.encoder = load_model(p / "encoder.keras")  # 또는 encoder.h5
        self.reg_next = joblib.load(p / "reg_next.pkl")
        self.object_ids = np.load(p / "object_ids.npy", allow_pickle=False)
        with (p / "meta.json").open() as f:
            self.meta: Dict[str, Any] = json.load(f)
        self.version = self.meta.get("modelVersion", "ttl-lstm-gbdt-unknown")

    def extract_embeddings(self, X_seq):
        return self.encoder.predict(X_seq, verbose=0)  # (n, 64)

    def predict_next_interval(self, emb):
        # train 때 log1p 사용했으면 여기서는 expm1 역변환 가정
        y_hat_log = self.reg_next.predict(emb)
        return np.expm1(y_hat_log)  # (n,)

def ttl_from_next_interval(next_sec: float,
                           min_ttl: int = 300, max_ttl: int = 86400) -> int:
    if not np.isfinite(next_sec) or next_sec <= 0:
        return min_ttl
    ttl = int(round(next_sec))
    return max(min_ttl, min(ttl, max_ttl))
