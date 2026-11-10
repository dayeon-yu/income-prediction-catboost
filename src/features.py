"""
소득 예측 프로젝트 — 전처리 · 피처 엔지니어링 모듈
code.ipynb의 파이프라인과 동일한 로직을 함수로 분리한 것.
모든 통계는 fit(train) → transform(test) 원칙을 지킨다 (데이터 누수 방지).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

TARGET = "income"
ID_COL = "id"
CAT_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
TE_COLS = ["occupation", "native_country", "edu_occ", "sex_marital", "workclass"]
AGG_SPECS = [
    ("occupation", ["age", "education_num", "hours_per_week"]),
    ("native_country", ["education_num", "hours_per_week"]),
    ("workclass", ["age", "hours_per_week"]),
]


def load_data(train_path: str, test_path: str):
    """CSV 또는 XLSX를 확장자로 판별해 로드한다."""
    reader = pd.read_excel if str(train_path).endswith((".xlsx", ".xls")) else pd.read_csv
    train_df = reader(train_path)
    test_df = reader(test_path)
    return train_df, test_df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """결측 처리 + 문자열 정규화.

    - 결측은 '?' 표기든 NaN(빈 칸)이든 모두 'Unknown' 카테고리로 **보존**한다.
      (workclass 결측 그룹의 >50K 비율 9.1% vs 정상 24.8% → 결측 자체가 신호)
    - 주의: fillna를 astype(str)보다 먼저 해야 한다. pandas 2.x에서 NaN.astype(str)은
      'nan' 문자열이 되어 Unknown으로 합쳐지지 않는다.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) in ("string", "str"):
            s = df[col].fillna("Unknown").astype(str).str.strip()
            s = s.replace({"?": "Unknown", "": "Unknown", "nan": "Unknown"})
            df[col] = s
    return df


def base_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Base FE 6개 (stateless): 행 단위로만 계산되므로 train/test에 같은 함수를 적용한다."""
    df = clean(df)
    df["cap_net"] = df["capital_gain"] - df["capital_loss"]
    df["has_capital_stats"] = ((df["capital_gain"] > 0) | (df["capital_loss"] > 0)).astype(int)
    df["gain_per_hour"] = df["capital_gain"] / (df["hours_per_week"] + 1)
    df["age_bins"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 60, 100],
        labels=["Young", "Junior", "Senior", "Veteran", "Retired"],
    ).astype(str)
    df["sex_marital"] = df["sex"] + "_" + df["marital_status"]
    df["edu_occ"] = df["education"] + "_" + df["occupation"]
    return df


def kfold_target_encode(train_series, target, test_series, n_splits=5, smoothing=10, random_state=42):
    """KFold Out-Of-Fold target encoding.

    train: 각 fold는 나머지 fold의 타겟 평균으로만 인코딩 (자기 자신의 타겟을 보지 않음)
    test : 전체 train 통계로 인코딩
    smoothing: 표본이 적은 카테고리는 global mean 쪽으로 수축(shrinkage)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    global_mean = target.mean()
    encoded_train = pd.Series(np.nan, index=train_series.index, dtype=float)

    def _smooth(x, y):
        agg = y.groupby(x).agg(["mean", "count"])
        return (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)

    for tr_idx, va_idx in kf.split(train_series):
        smooth = _smooth(train_series.iloc[tr_idx], target.iloc[tr_idx])
        encoded_train.iloc[va_idx] = train_series.iloc[va_idx].map(smooth).fillna(global_mean).values

    smooth_full = _smooth(train_series, target)
    encoded_test = test_series.map(smooth_full).fillna(global_mean)
    return encoded_train, encoded_test


def add_target_encoding(X_tr, y_tr, X_te, cols=TE_COLS, **kw):
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col in cols:
        enc_tr, enc_te = kfold_target_encode(X_tr[col], y_tr, X_te[col], **kw)
        X_tr[f"{col}_te"] = enc_tr.values
        X_te[f"{col}_te"] = enc_te.values
    return X_tr, X_te


def add_group_aggregations(X_tr, X_te, specs=AGG_SPECS):
    """그룹별 평균(cross-row 통계)을 row-level feature로 추가. 통계는 train에서만 계산."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for group_col, num_cols in specs:
        stats = X_tr.groupby(group_col)[num_cols].mean()
        for col in num_cols:
            new_col = f"{group_col}_mean_{col}"
            gmean = X_tr[col].mean()
            X_tr[new_col] = X_tr[group_col].map(stats[col]).fillna(gmean)
            X_te[new_col] = X_te[group_col].map(stats[col]).fillna(gmean)
    return X_tr, X_te


def build_features(train_df, test_df, *, target_encoding=True, group_agg=True, random_state=42):
    """원본 → 31개 변수. 반환: X_train, y_train, X_test, test_ids, cat_features"""
    y = (train_df[TARGET].astype(str).str.strip() == ">50K").astype(int)
    test_ids = test_df[ID_COL]
    X = base_engineering(train_df.drop(columns=[TARGET, ID_COL]))
    X_test = base_engineering(test_df.drop(columns=[ID_COL]))
    if target_encoding:
        X, X_test = add_target_encoding(X, y, X_test, random_state=random_state)
    if group_agg:
        X, X_test = add_group_aggregations(X, X_test)
    cat_features = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("string", "str")]
    return X, y, X_test, test_ids, cat_features


def build_features_split_first(train_df, val_df, *, random_state=42):
    """검증용: train/val을 먼저 나눈 뒤 train 통계로만 val을 인코딩한다 (더 엄격한 누수 차단)."""
    tr = train_df.rename(columns={}).copy()
    va = val_df.copy()
    y_tr = (tr[TARGET].astype(str).str.strip() == ">50K").astype(int)
    y_va = (va[TARGET].astype(str).str.strip() == ">50K").astype(int)
    X_tr = base_engineering(tr.drop(columns=[TARGET, ID_COL]))
    X_va = base_engineering(va.drop(columns=[TARGET, ID_COL]))
    X_tr, X_va = add_target_encoding(X_tr, y_tr, X_va, random_state=random_state)
    X_tr, X_va = add_group_aggregations(X_tr, X_va)
    cat_features = [c for c in X_tr.columns if X_tr[c].dtype == object or str(X_tr[c].dtype) in ("string", "str")]
    return X_tr, y_tr, X_va, y_va, cat_features
