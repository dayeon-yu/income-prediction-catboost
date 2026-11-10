# 📁 데이터 안내

본 폴더에는 **원본 데이터셋이 포함되어 있지 않습니다.**
데이터는 강좌(Pattern Recognition)에서 배포된 자료로, 재배포하지 않습니다.

## 필요한 파일

`code.ipynb`와 `notebooks/analysis.ipynb` 실행을 위해 아래 두 파일을 이 폴더(`data/`)에 배치하세요.

| 파일 | 규모 | 설명 |
| --- | --- | --- |
| `train.csv` | 39,073 rows × 15 cols | 학습용 (타겟 `income` 포함) |
| `test.csv` | 9,769 rows × 14 cols | 예측용 (타겟 미포함, `id` 유지) |

- 결측은 빈 칸(NaN)으로 들어 있습니다 (`workclass` 2,213 · `occupation` 2,223 · `native_country` 678).
  코드는 `?` 표기와 NaN을 모두 `Unknown` 카테고리로 통일합니다.
- 팀 작업 당시에는 같은 데이터를 `train.xlsx` / `test.xlsx`로 썼습니다. 노트북은 확장자로 CSV/XLSX를 자동 판별합니다.

## 경로 설정

`code.ipynb` 상단의 경로 변수를 환경에 맞게 수정하세요.

```python
TRAIN_PATH = 'data/train.csv'
TEST_PATH  = 'data/test.csv'
```

> Google Colab 사용 시에는 `drive.mount()` 후 Google Drive 경로를 지정합니다.
