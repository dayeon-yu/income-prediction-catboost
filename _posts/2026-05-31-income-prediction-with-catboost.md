---
layout: post
title: "CatBoost로 소득 예측하기 — 피처를 '더하지 않는' 전략"
date: 2026-05-31 12:00:00 +0900
categories: [Machine Learning, Project]
tags: [CatBoost, Feature Engineering, Target Encoding, Pattern Recognition, 이진분류]
---

> Pattern Recognition 최종 팀프로젝트(Team 16) 회고.
> 인구통계 데이터로 **연소득 $50K 초과 여부**를 예측하며 배운 것들을 정리함.

## 🎯 문제

- **목표**: 개인의 인구통계·근로 정보로 `income > 50K` 여부를 맞히는 **이진 분류**
- **데이터**: train 39,073행 × 15열 / test 9,769행 × 14열
- **평가 지표**: `(F1 + AUC) / 2` — 불균형 데이터라 정확도만으로는 부족
- **난관**: `<=50K` 76% vs `>50K` 24%의 **약 3:1 클래스 불균형**

## 🔍 EDA에서 얻은 힌트

데이터를 뜯어보며 "선형보다 조건부 규칙이 강한 데이터"라는 감을 잡음.

- `education_num`이 소득과 상관 0.335로 가장 높음 (박사급 고소득 비율 70%대)
- `capital_gain = 99,999`인 206명은 **전원 고소득** → 강한 규칙 존재
- 혼인상태 `Married-civ-spouse` 44%, `Never-married` 4.6% → 극단적 편차
- 직업군 `Exec-managerial` 48% vs `Priv-house-serv` 1.5%

➡️ **결론**: 특정 조건이 충족될 때 클래스가 명확히 갈리는 **비선형 구조** → 트리 기반 부스팅이 적합.

## 🧠 왜 CatBoost인가

- 범주형 변수(`workclass`, `occupation`, `native_country` 등)가 매우 많음
- CatBoost는 범주형을 **내부적으로 직접 인코딩** → 원-핫 폭증 없이 처리
- 결측을 별도 카테고리로 다룰 수 있어 "결측 자체가 신호"인 상황에 유리

## 💡 핵심 인사이트 — "더하지 않는" 피처 엔지니어링

이번 프로젝트의 가장 큰 교훈. **모든 파생 변수가 도움이 되는 건 아니다.**

CatBoost는 이미 강력하기 때문에, 아래처럼 모델이 스스로 학습 가능한 것들은 추가해도 **정보 이득이 0**이었음.

- StandardScaler → 트리 모델은 스케일 불변
- Winsorize / log1p → 단조변환에 불변
- `is_married` 같은 단순 binary → CatBoost가 이미 학습 가능

대신 **CatBoost가 단일 행만으로는 학습할 수 없는 신호**만 선별해서 추가함.

| 유형 | 추가한 것 | 이유 |
| --- | --- | --- |
| Target Encoding (OOF) | 카테고리별 평균 타겟값 | 내부 인코딩과 다른 신호 |
| Group Aggregation | '같은 직업군 평균 나이' 등 | cross-row 통계는 단일 행으로 불가 |

결과적으로 원본 **13개 → 31개** 변수로만 확장. 무작정 늘리지 않은 게 핵심.

### 누수 방지가 생명

- **Target Encoding**은 5-fold **Out-Of-Fold**로 계산 (각 fold는 나머지 fold의 통계만 사용)
- 모든 통계는 **train에서만 fit**, test는 transform만
- 검증: train/test 인코딩 분포 평균 차이 **0.003 이내** → 누수 없음 확인

## ⚙️ 모델링 & Threshold 튜닝

```python
CatBoostClassifier(
    iterations=2500, learning_rate=0.02, depth=7,
    l2_leaf_reg=10, bagging_temperature=0.8,
    random_strength=1, border_count=254,
    early_stopping_rounds=200, random_seed=42
)
```

- 작은 learning rate + 많은 iteration + early stopping으로 안정적 수렴
- 평가 지표가 `(F1+AUC)/2`이므로, 기본 threshold 0.5를 그대로 쓰지 않고 **0.1~0.8을 탐색**
- **최적 threshold = 0.4220** 선택

## 📊 결과

| 지표 | 값 |
| --- | --- |
| Validation AUC | 0.9306 |
| Validation F1 | 0.7373 |
| **(F1 + AUC) / 2** | **0.8339** |

- Test 예측: ≤50K 7,474명 / >50K 2,295명 → 편향 없는 안정적 분포
- 전체 train 재학습 후 threshold 적용해 `prediction.csv` 생성

## 🌱 배운 점

- **피처는 많다고 좋은 게 아니다** — 모델이 못 배우는 것만 골라 더하기
- **누수 방지는 습관** — fit은 train에서만, OOF로 검증
- **지표에 맞춘 최적화** — 평가식이 `(F1+AUC)/2`면 threshold도 거기에 맞춰야 함

---

*코드와 보고서 전문은 [GitHub 저장소](https://github.com/dayeon-yu/income-prediction-catboost)에서 확인할 수 있습니다.*
