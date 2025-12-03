from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, validator
import uuid, asyncio
from typing import Literal
from .csv_logger import SingleCSVLogger
from .feature_builder import build_last_window_from_csv
from .artifacts_loader import Artifacts, ttl_from_next_interval
import numpy as np
import traceback

CSV_PATH = "data/all_events.csv"
ART_DIR  = "artifacts"

app = FastAPI(title="TTL Model API", version="1.0.0")
logger = SingleCSVLogger(CSV_PATH)
art: Artifacts | None = None

class ModelRequestDTO(BaseModel):
    objectId: int | str
    requestTime: int = Field(ge=0)
    sizeBytes: int = Field(ge=0)

    @validator("requestTime")
    def _epoch_sane(cls, v):
        if v < 978307200 or v > 4102444800:
            raise ValueError("requestTime seems out of range (epoch seconds)")
        return v

class ModelResponseDTO(BaseModel):
    ttlSec: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    modelVersion: str
    requestId: str

@app.on_event("startup")
async def load_artifacts():
    global art
    art = Artifacts(ART_DIR)

@app.get("/health")
async def health():
    return {"status": "ok", "version": art.version if art else None}

@app.post("/v1/ttl", response_model=ModelResponseDTO)
async def infer_ttl(req: ModelRequestDTO):
    if art is None:
        raise HTTPException(status_code=503, detail="model_not_loaded")

    req_id = uuid.uuid4().hex

    try:
        m = int(art.meta["m"]); k = int(art.meta["k"])
        window_size = int(art.meta["window_size"])

        # 최근 이벤트 기반 feature 전체 (모든 object에 대해)
        X_last_all, id2idx = build_last_window_from_csv(
            CSV_PATH, art.object_ids, m, k, window_size, 1
        )

        # 아직 학습할 만큼 데이터가 없는 경우 → 보수적인 TTL
        if X_last_all.shape[0] == 0:
            ttl = 300
            conf = 0.2
        else:
            # 요청 objectId를 int로 변환 (art.object_ids도 int일 것으로 가정)
            try:
                oid = int(req.objectId)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid_object_id")

            if oid not in id2idx:
                # 모델이 모르는 object → 보수적인 기본값
                ttl = 300
                conf = 0.2
            else:
                idx = id2idx[oid]

                # 해당 object 한 줄만 뽑기: (1, m, feature)
                x_single = X_last_all[idx : idx + 1, :, :]  # (1, m, 2)

                # LSTM이 1채널만 기대하면 첫 채널만 사용
                if x_single.shape[-1] > 1:
                    x_single = x_single[..., :1]   # (1, m, 1)

                x_single = x_single.astype("float32")

                # 3) 임베딩 → 다음 간격 예측 (해당 object에 대해서만)
                emb = art.extract_embeddings(x_single)          # (1, 64) 예상
                next_arrival = float(art.predict_next_interval(emb)[0])  # seconds

                # 4) object별 TTL 계산
                ttl = ttl_from_next_interval(
                    next_arrival,
                    min_ttl=1,
                    max_ttl=86400
                )
                conf = 0.7

        resp = ModelResponseDTO(
            ttlSec=ttl,
            confidence=conf,
            modelVersion=art.version,
            requestId=req_id
        )

        await logger.append({
            "request_id": req_id,
            "object_ID": req.objectId,
            "request_time": req.requestTime,
            "size_bytes": req.sizeBytes,
        })

        return resp

    except Exception as e:
        print("==== Inference Failed TraceBack ====")
        traceback.print_exc()
        print("Exception object:", repr(e))
        raise HTTPException(status_code=500, detail=f"inference_failed: {e}")

