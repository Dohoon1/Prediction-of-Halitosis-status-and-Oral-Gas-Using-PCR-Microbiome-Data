# PCR major-revision analysis code (2026-10-21)

## 실행 단계와 완료 의미

- `preflight`: 16종 whitelist, 102명/72·30, ID-라벨 해시, VSC 합계, 별도 Total 제외를 검증합니다.
- `descriptive`: 실제 16종 자료의 다양성, raw/log/relative/CLR 군 비교, 효과크기·보정, 상관·군집 안정성, 자료 inventory를 계산합니다. 모델은 학습하지 않습니다.
- `analysis`: 기술통계와 본 반복 nested CV, OOF, 조건부 participant-block CI, clinical 대비 paired Δ, 절대오차 비교를 계산합니다. 순열 100회와 모든 sensitivity 학습을 자동으로 함께 실행하지 않습니다.
- `sensitivity --sensitivity-models`: 명시된 모든 모델 민감도 프로필과 실행 가능 대체 outcome을 별도 폴더에서 평가합니다. runner 없이 실행하면 정의·자료·실행계획만 저장하고 미학습을 표시합니다.
- `permutation`: 같은 5×2 절차로 관측 기준을 다시 학습하고 B=100 순열을 진행합니다. 체크포인트에서 중단·재개할 수 있습니다. feature importance의 3회 셔플과 다른 검정입니다.

`manifest.status=complete`는 선택한 단계가 끝났다는 뜻입니다. descriptive complete는 본 모델, sensitivity 또는 순열이 끝났다는 뜻이 아닙니다. 실행된 단계·개발 프로필 여부·config/입력/코드 해시를 확인하십시오.

## 분석 규칙

| 항목 | 규칙 |
|---|---|
| 원래 outcome | 기존 Halitosis 72/30 유지; sex-specific threshold와 6명 차이는 audit에 남김 |
| 총량 | PCR_panel_total = 16종 농도 합계; Excel 별도 Total bacteria는 입력에서 제외 |
| 다양성 | 검출 표적 수, Shannon 자연로그, Pielou; Chao1 제외. 전체 미생물 다양성으로 일반화하지 않음 |
| 본 CV | outer 5-fold × 10회; inner 3-fold; outer_repeats=20으로 변경 가능 |
| 분할 | 분류 층화, 회귀 repeated KFold. outer/inner test 정보로 전처리·군집·튜닝하지 않음 |
| sampling | train imputation/표준화/범주코딩 → SMOTENC → 최종 one-hot. none 및 train class_weight sensitivity |
| 군집 | r>0.4 positive Pearson connected components. train largest/second/union; 연결성분은 clique가 아님 |
| 추가 축약 | train 최대 군집의 log1p 평균 / train 표준화·PCA의 PC1. 원 고정 Cluster 1과 동일 구성원이라는 뜻이 아님 |
| 비교용 군집 | 전체 cohort 고정 군집은 별도 exploratory sensitivity |
| 선택 | 종/알고리즘/파라미터의 inner 선택을 outer test에서 평가하는 Nested_best_single/candidate |
| 주 성능 | 반복별 pooled OOF 성능의 평균. 사람별 예측을 평균한 뒤 구한 성능은 별도 열 |
| CI·Δ | 사람 단위 block bootstrap; 모든 반복과 paired 모델이 같은 사람에 묶임. fixed OOF 조건부이며 재학습/선택 불확실성을 포함하지 않음 |
| 회귀 | raw-ppb R2, log-target R2, ppb MAE/RMSE, train-mean 대비 skill을 구분 |
| 회귀 검정 | 사람별 반복 평균 absolute error의 Friedman/Wilcoxon; gas/scale별 Holm. R2 비교 검정이라고 부르지 않음 |
| 순열 | B=100; 관측·null 모두 outer5×2/inner3. 순열 라벨별 층화 재생성. 가스는 같은 대상자 순열로 함께 셔플 |
| 최대값 검정 | 명시된 전체 후보 family 내 최대 AUROC/R2; 가스별 및 가스 공동 family. p=(초과·동률+1)/(B+1) |
| 표기 | 계산 full precision; performance 표시 2자리. p는 별도 3자리 또는 <0.001 |

추가 모델/축약/selector 때문에 원고의 이전 모델 수를 그대로 쓸 수 없습니다. 실행 후 `model_inventory.csv`, `reported_model_inventory.csv`, `methods_inventory.csv`에서 실제 후보와 모델 수를 확인합니다. selector는 이미 학습한 후보 중 inner 점수만으로 고른 예측을 재사용합니다.

## 명령 예시 (PowerShell)

Python 환경은 numpy/pandas/openpyxl/scipy/sklearn/imblearn/torch/matplotlib가 설치되어 있어야 합니다. 이번 검증에는 기존 PCR 프로젝트의 `.venv/Scripts/python.exe`를 사용했습니다. 새 폴더에 기존 `.venv`를 복사하거나 설치를 변경하지 않았습니다. 일반 의존성은 requirements.txt, 선택형 보정 연관성 분석 의존성은 requirements-optional.txt입니다.

```powershell
$py = 'C:/Users/IEJD_HOMEPC/Desktop/PCR_revision/pcr_halitosis_restructured/.venv/Scripts/python.exe'
& $py code/01_analysis.py --config config/revision_main.json --stage preflight --results results/preflight_new
& $py code/01_analysis.py --config config/revision_main.json --stage descriptive --results results/descriptive_new
& $py code/01_analysis.py --config config/revision_main.json --stage analysis --results results/main_new
& $py code/02_figures.py --results results/main_new --output figures/main_new --dpi 300 --formats png tiff
& $py code/01_analysis.py --config config/revision_main.json --stage sensitivity --sensitivity-models --results results/sensitivity_new
```

각 run은 새 빈 폴더를 사용합니다. 기존 결과를 덮어쓰지 않습니다. `run_revision.ps1`도 같은 실행을 도와줍니다. full main/sensitivity는 많은 연산을 포함합니다.

```powershell
# 요청 B=100을 유지하고 이번 실행에서는 새 순열 10개까지만 진행합니다.
& $py code/01_analysis.py --config config/revision_permutation.json --stage permutation --results results/permutation_new --permutation-batch-size 10
# 동일 input/config/code일 때만 재개합니다. 분석 정의를 바꾼 뒤 이어 붙이지 않습니다.
& $py code/01_analysis.py --config config/revision_permutation.json --stage permutation --results results/permutation_new --permutation-batch-size 10 --resume
# B가 모두 끝나면 최종 null 그림을 생성합니다.
& $py code/03_permutation_figures.py --results results/permutation_new --output figures/permutation_new
```

`revision_integration_check.json`은 2-fold×2, 짧은 epoch/작은 grid/B=3으로 실행 연결을 확인하는 개발 프로필입니다. 여기서 나온 성능·p·CI를 원고에 쓰면 안 됩니다. B=100 프로필과 결과를 합치지 않습니다. tests/의 모의 runner는 통계·재개 동작을 검증하며 실제 연구 모델의 대체물이 아닙니다.

## 자료/판정이 필요한 부분

- 단일 total VSC, H2S, CH3SH cutoff는 값·단위·장비/문헌 근거를 지정해야 실행합니다. 빈 값을 임의로 채우지 않습니다. sex upper quartile은 cohort-relative high-gas라는 별도 정의이며 검증된 진단 기준으로 표현하지 않습니다.
- 반복 VSC가 없으면 ICC/CV를 계산하지 않습니다. 파일이 있으면 participant_id, analyte, visit, replicate, value_ppb와 반복설계를 명시해야 합니다. 기본 지원은 balanced common-occasion two-way random absolute-agreement ICC(2,1)이며 다른 설계에 그대로 적용하지 않습니다.
- PD/BOP/설태/타액유량은 PCR cohort에 실제 존재하는 metadata만 ID 일대일 확인 후 사용합니다. 다른 연구의 임상값을 가져오지 않습니다.
- 현재 환경에는 선택 의존성 statsmodels가 없어 보정 연관성 회귀는 not_run으로 남습니다. 설치 후 실제 공변량과 수렴/분리 진단을 확인해야 합니다. Firth는 구현되지 않았으며 일반 Logit을 Firth로 표시하지 않습니다.
- 원본 고정 epoch를 유지하므로 early stopping은 false로 기록합니다. true를 요청하면 지원되지 않는 설정으로 거부합니다. Transformer seed sensitivity는 별도 학습 결과를 필요로 합니다.
- Organoleptic/DMS/임상 반복측정의 부재, cutoff의 임상 타당성, 모집 근거는 코드만으로 해결되지 않습니다.

## 파일 구조

- common.py: 입력·코호트·라벨·표적 검증.
- pcr_statistics.py: 패널 지표, 다중검정, 효과크기, 정규화/상관/군집 안정성, 선택형 연관성.
- modeling.py / neural.py: 반복 nested 학습, fold별 전처리/SMOTENC, 후보·selector, NN 학습.
- validation.py: paired CI, 사람 단위 오차 검정, label-permutation/max statistic 및 재개.
- sensitivity.py: 사전 지정 민감도, 원래 outcome과 별도 대체 outcome, 자료·신뢰도 inventory.
- reporting.py: 실제 설정/모델 수, Methods·Word용 표시 CSV·legend·복잡도·진단 내보내기.
- 01_analysis.py: 단계 실행과 manifest; 02/03: 저장 결과만 그림으로 렌더링.
- review/: 변경·코멘트 대응·검증·원본 해시. 원고/표/리뷰탈 Word는 이 코드 수정 작업에서 변경하지 않았습니다.

같은 panel total을 공변량으로 보정하면 part-whole coupling이 생깁니다. panel-relative/CLR/partial correlation이 독립 전체 미생물 총량 보정의 증거라고 주장하면 안 됩니다. global outcome permutation은 clinical 대비 PCR의 조건부 추가 효과 검정이 아닙니다. 추가 기여는 동일 split의 paired clinical 비교로 별도 보고합니다.


## 최종 검증과 formal family 범위

최종 코드의 테스트 43개가 통과했습니다. 실제102명 축소통합 실행은 분류13개/회귀39개이고 모든 모델당102명×2반복 OOF입니다. CI·paired Δ·회귀 오차 검정의 참여자 단위가102명임을 검산했습니다. B3 실제 CLI는 첫배치1개에서 incomplete/최종p미표시, 재개2개 후 complete를 확인했습니다. 이는개발 검증이며 본5×10/B1000 결과가 아닙니다.

최대통계 family는 해당 프로필에 선언된 non-Dummy 후보와 nested selector입니다. Dummy는 대조이므로 max 검색에서 제외되고 별도 null 대조를 가집니다. 다른 normalization/cutoff/대체outcome/seed 프로필은 사전지정 민감도로 모두 공개하며, 그중 가장 좋은 값을 고르는 검정은 이 main family p로 보정됐다고 주장하지 않습니다.


## 마지막 메모 대조에서 추가한 산출물

- `dummy_reference_deltas.csv`: clinical 대비 주 ΔCI와 분리한 Dummy 대비 참조 ΔCI. Clinical_only와 nested selector도 포함하며 PCR의 조건부 추가 기여로 해석하지 않습니다.
- `outcome_sensitivity_summary.csv`: 원래 기준/대체 기준별 양성 수, E. nodatum log10(copies+1) 평균 차이·Cliff delta·pointwise CI, 학습된 경우의 탐색적 최대 AUROC와 nested selector AUROC. 미학습 값은 빈칸/미수행입니다.
- 대체 기준 효과의 다중검정은 동일 label을 만드는 중복 정의를 하나로 묶은 기준별 단일 사전지정 E. nodatum family이며, 주 분석의 16종 family와 다릅니다. OR 또는 보정된 인과 효과를 만들었다고 표시하지 않습니다.
- `review/Word_definition_change_locations.csv`: Word에서 용어·정의·기존 다양성 수치를 검토할 XML 문단 위치. 원고 본문이 포함되어 코드 전용 ZIP에는 넣지 않습니다.


최종 검수: 2026-10-22. Transformer seed 민감도 완료 결과는 transformer_seed_results.csv와 transformer_seed_summary.csv에서 mean/SD와 실제 완료 seed 수로 집계합니다. 다른 normalization/군집 조건을 섞거나 CI 경계를 평균하지 않습니다. 아직 해당 학습을 하지 않은 run은 seed inventory에 not_run으로 표시합니다.
