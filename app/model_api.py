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
        # 1) 단일 CSV에 먼저 기록(모델 추론 결과도 기록해야 하므로, 이번에는 사후에 한 번 더 업데이트 X → 한번에 기록)
        #    여기서는 모델 응답이 생성된 뒤 한번에 기록합니다.

        # 2) 최근 윈도우에서 특징 생성
        m = int(art.meta["m"]); k = int(art.meta["k"])
        window_size = int(art.meta["window_size"]); step = int(art.meta["step"])
        X_last, id2idx = build_last_window_from_csv(CSV_PATH, art.object_ids, m, k, window_size, step)

        if X_last.shape[0] == 0:
            # 아직 데이터가 충분치 않으면 보수적 TTL 반환
            ttl = 300
            conf = 0.2
        else:
            # 2) LSTM이 기대하는 feature 차원 맞추기 (feature가 2개 이상이면 첫 번째만 사용)
            if X_last.shape[-1] > 1:
                X_last = X_last[..., :1]  # (batch, m, 1)로 슬라이스
            X_last = X_last.astype("float32")

            # 3) 임베딩 → 다음 간격 예측
            emb = art.extract_embeddings(X_last)  # (1, 64)
            next_arrival = float(art.predict_next_interval(emb)[0])  # seconds

            # 4) 요청 objectId에 초점을 맞춘 TTL
            ttl = ttl_from_next_interval(next_arrival, min_ttl=1, max_ttl=86400)
            conf = 0.7  # 가벼운 기본값. calibration을 붙였다면 여기 반영

        resp = ModelResponseDTO(
            ttlSec=ttl,
            confidence=conf,
            modelVersion=art.version,
            requestId=req_id
        )

        # 5) 최종 로그 기록
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
