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
    for i in range(0, len(df) - total, step):
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
    서빙용: 최근 한 윈도우 묶음에서 X_seq 하나만 생성해 돌려줍니다.
    return: (X_seq_last (1, m, 1), idx_map(object_id -> index))
    """    
    total_needed = window_size * (m + k) + step  # 여유
    df = read_tail_for_windows(csv_path, total_needed)
    print("df length:", len(df))
    print("df.columns:", df.columns)
    print(df.head())

    # CSV에는 object_ID로 와도, 내부에서는 object_id로 통일
    if "object_ID" in df.columns:
        df = df.rename(columns={"object_ID": "object_id"})
    if "objectId" in df.columns:
        df = df.rename(columns={"objectId": "object_id"})

    if "request_time" in df.columns:
        print("request_time range:",
            df["request_time"].min(),
            df["request_time"].max())

    # 2. 유효한 object_id만 필터
    valid_ids = set(int(x) for x in object_ids)
    if "object_id" in df.columns:
        df = df[df["object_id"].isin(valid_ids)]

    # 3. 디버깅용 출력은 이제 전부 object_id 기준으로
    if "object_id" in df.columns:
        print(df["object_id"].value_counts())
    if df.empty or len(df) < window_size * (m + k):
        return np.zeros((0, m, 1)), {oid: i for i, oid in enumerate(object_ids)}

    X_seq, _, _ = build_simple_sequences(df, object_ids, m, k, window_size, step, time_col="request_time")
    if X_seq.shape[0] == 0:
        return np.zeros((0, m, 1)), {oid: i for i, oid in enumerate(object_ids)}

    X_last = X_seq[-1:,:,:]  # 맨 마지막 배치 1개만 사용
    id2idx = {o: i for i, o in enumerate(object_ids)}
    return X_last, id2idx