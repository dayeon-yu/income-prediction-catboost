"""평가 유틸: 과제 지표 (F1 + AUC) / 2, threshold 탐색, 그룹별 공정성 지표."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def project_metric(y_true, y_prob, threshold=0.5):
    """과제 공식 평가 지표 (F1 + AUC) / 2 와 구성 요소를 dict로 반환."""
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, (np.asarray(y_prob) >= threshold).astype(int))
    return {"AUC": auc, "F1": f1, "(F1+AUC)/2": (f1 + auc) / 2, "threshold": threshold}


def best_threshold(y_true, y_prob, grid=None):
    """(F1+AUC)/2 를 최대화하는 threshold. AUC는 threshold와 무관하므로 F1 최대화와 동일."""
    grid = np.linspace(0.1, 0.8, 701) if grid is None else grid
    f1s = np.array([f1_score(y_true, (y_prob >= t).astype(int)) for t in grid])
    i = int(f1s.argmax())
    return float(grid[i]), float(f1s[i]), grid, f1s


def group_fairness(y_true, y_prob, group, threshold=0.5):
    """그룹(성별·인종 등)별 예측 양성률, TPR(recall), FPR, 정밀도, F1, AUC."""
    y_true = np.asarray(y_true); y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    rows = []
    for g in pd.Series(group).unique():
        m = np.asarray(group) == g
        if m.sum() < 30:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true[m], y_pred[m], labels=[0, 1]).ravel()
        auc = roc_auc_score(y_true[m], np.asarray(y_prob)[m]) if len(set(y_true[m])) > 1 else np.nan
        rows.append({
            "group": g, "n": int(m.sum()), "actual_pos_rate": y_true[m].mean(),
            "pred_pos_rate": y_pred[m].mean(),
            "TPR(recall)": tp / (tp + fn) if tp + fn else np.nan,
            "FPR": fp / (fp + tn) if fp + tn else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "F1": f1_score(y_true[m], y_pred[m]) if tp + fn else np.nan,
            "AUC": auc,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
