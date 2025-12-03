import numpy as np
import pandas as pd
from typing import Tuple

def read_tail_for_windows(csv_path: str, total_needed: int) -> pd.DataFrame:
    # 성능 단순화를 위해 전체 읽고 tail 하는 방식(프로토타입).
    # 운영에선 pyarrow/duckdb/head-tail index/파티셔닝 고려 권장.
    df = pd.read_csv(csv_path)
    if len(df) > total_needed:
        df = df.tail(total_needed)
    return df


def build_simple_sequences(df, object_ids, m, k, window_size, step, time_col="request_time"):
    """
    df: time-sorted DataFrame with [time_col, 'object_ID']
    object_ids: unique object IDs (sorted)
    return: X_seq (N, m, 1), y_pop (N, d), y_next (N, d)
    """
    X_seq, y_pop, y_next = [], [], []

    # 1) 시간 정렬
    df = df.sort_values(time_col).reset_index(drop=True)

    # 2) 시간 배열/객체 배열 만들기
    ts = df['request_time'].to_numpy().astype(float)  # seconds (float)
    obj = df['object_id'].to_numpy()
    id2idx = {o: i for i, o in enumerate(object_ids)}
    d = len(object_ids)

    # 3) 슬라이딩 윈도우
    total = window_size * (m + k)
    for i in range(0, len(df) - total + 1, step):
        seq = obj[i : i + total]
        ts_seq = ts[i : i + total]

        # 입력: 과거 m개 창 (각 창 길이: window_size)
        x_seq = []
        for j in range(m):
            w = seq[j*window_size : (j+1)*window_size]
            ts_w  = ts_seq[j*window_size : (j+1)*window_size]
            
            counts = np.zeros(d, dtype=float)
            unique, cnts = np.unique(w, return_counts=True)
            for u, c in zip(unique, cnts):
                counts[id2idx[u]] = c / window_size      # 비율

            # 평균 간격 — 창 전체의 평균 간격을 각 객체 위치에 broadcast
            if len(ts_w) >= 2:
                gap_mean = float(np.mean(np.diff(ts_w)))
            else:
                gap_mean = float(window_size)
            gap_mean = 0.0 if not np.isfinite(gap_mean) else gap_mean
            gap_vec = np.full_like(counts, gap_mean, dtype=float)

            # (d, 2)
            x_seq.append(np.stack([counts, gap_vec], axis=1))
            
        # (m, d) -> (d, m, 1)
        # X_seq.append(np.array(x_seq).T.reshape(-1, m, 1))
        X_seq.append(np.stack(x_seq, axis=0).transpose(1, 0, 2))  # (d, m, 2)

        # 미래 구간
        future_objs = seq[m*window_size:]
        future_ts   = ts_seq[m*window_size:]

        # 인기도 라벨: 미래 구간에 한 번이라도 등장했는지
        L = 3
        future_subset = future_objs[:L * window_size]
        y_pop.append(np.isin(object_ids, future_subset).astype(int))

        
        # y_pop.append(np.isin(object_ids, future_objs).astype(int))  # (d,)

        # (b) 다음 inter-arrival: 기준시각 이후 첫 등장까지 시간
        next_time = []
        t_ref = ts_seq[m*window_size - 1]
        for oid in object_ids:
            mask = (future_objs == oid)
            if np.any(mask):
                idx_first = np.argmax(mask)  # True가 처음인 위치
                next_time.append(max(0.0, float(future_ts[idx_first] - t_ref)))
            else:
                next_time.append(np.nan)     # 미래에 아예 안 나옴
        y_next.append(next_time)

    if not X_seq:
        return (np.zeros((0, m, 1)), np.zeros((0, d), int), np.zeros((0, d), float))

    X_seq = np.concatenate(X_seq, axis=0)   # (N, m, 1)
    y_pop = np.array(y_pop, dtype=int)      # (N, d)
    y_next = np.array(y_next, dtype=float)  # (N, d)
    return X_seq, y_pop, y_next


def build_last_window_from_csv(csv_path: str,
                               object_ids: np.ndarray,
                               m: int, k: int,
                               window_size: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    서빙용: 가장 최근의 m개 윈도우(과거 데이터)만 사용하여 X_seq를 생성합니다.
    미래 데이터(k)를 기다리지 않고, 현재 시점의 예측을 수행합니다.
    return: (X_seq_last (1, m, 1), idx_map)
    """    
    # 필요한 과거 데이터 길이: m * window_size
    # (step은 여기서 무시하고 무조건 가장 최근 데이터를 봅니다)
    past_needed = m * window_size
    
    # 여유있게 읽기
    df = read_tail_for_windows(csv_path, past_needed + 50)
    
    # 1. 컬럼 정리
    if "object_ID" in df.columns:
        df = df.rename(columns={"object_ID": "object_id"})
    if "objectId" in df.columns:
        df = df.rename(columns={"objectId": "object_id"})
        
    # 2. 유효 object_id 필터
    valid_ids = set(int(x) for x in object_ids)
    if "object_id" in df.columns:
        df = df[df["object_id"].isin(valid_ids)]
        
    # 3. 데이터 부족 시 처리
    if len(df) < past_needed:
        # 데이터가 모자라면 0으로 채워진 더미 반환 (혹은 에러 처리)
        id2idx = {o: i for i, o in enumerate(object_ids)}
        return np.zeros((0, m, 1)), id2idx

    # 4. 가장 최근 past_needed 개수만 취함
    df = df.tail(past_needed).sort_values("request_time")
    
    ts = df['request_time'].to_numpy().astype(float)
    obj = df['object_id'].to_numpy()
    
    id2idx = {o: i for i, o in enumerate(object_ids)}
    d = len(object_ids)
    
    # 5. Feature Building (X only)
    # m개의 윈도우를 순회하며 feature 생성
    x_seq = []
    for j in range(m):
        # j번째 윈도우 (0 ~ window_size, window_size ~ 2*window_size, ...)
        w_obj = obj[j*window_size : (j+1)*window_size]
        w_ts  = ts[j*window_size : (j+1)*window_size]
        
        counts = np.zeros(d, dtype=float)
        unique, cnts = np.unique(w_obj, return_counts=True)
        for u, c in zip(unique, cnts):
            if u in id2idx:
                counts[id2idx[u]] = c / window_size
        
        if len(w_ts) >= 2:
            gap_mean = float(np.mean(np.diff(w_ts)))
        else:
            gap_mean = float(window_size)
        gap_mean = 0.0 if not np.isfinite(gap_mean) else gap_mean
        gap_vec = np.full_like(counts, gap_mean, dtype=float)
        
        # (d, 2)
        x_seq.append(np.stack([counts, gap_vec], axis=1))
        
    # Stack -> (m, d, 2) -> (d, m, 2)
    X_out = np.stack(x_seq, axis=0).transpose(1, 0, 2)
    
    # Batch dimension 추가 -> (1, d, m, 2) ?? 
    # 기존 build_simple_sequences 반환값은 (N, m, 1) 형태였음 (ensure_3d_single_channel 전)
    # 하지만 train.py에서 ensure_3d_single_channel로 (N, m, 1)로 바꿈.
    # 여기서도 (1, m, 1) 형태로 맞춰줘야 함?
    # 아니, build_simple_sequences의 리턴은:
    # X_seq.append(np.stack(x_seq, axis=0).transpose(1, 0, 2)) -> (d, m, 2)
    # 그리고 np.concatenate(X_seq, axis=0) -> (N*d, m, 2) 가 아니라...
    # 아, build_simple_sequences 코드를 보면:
    # X_seq 리스트에 (d, m, 2)를 append 함.
    # 마지막에 np.concatenate(X_seq, axis=0) 하면 (N*d, m, 2)가 됨.
    # 즉, (배치크기, m, 2) 형태.
    
    # 우리는 배치크기=1 (사실상 d개의 object에 대한 1개의 시점)이 아니라
    # "현재 시점"에서의 d개 object 각각에 대한 feature가 필요함.
    # model_api.py에서는 X_last[..., :1] 로 슬라이싱해서 씀.
    # 즉 (d, m, 2)를 리턴하면 됨. (d가 배치 차원 역할)
    
    return X_out, id2idx