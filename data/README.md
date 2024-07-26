## Credit Card Fraud Detection Dataset

이 데이터셋은 2013년 9월에 유럽에서 신용카드를 소지한 고객들이 수행한 신용카드 거래를 포함하고 있습니다. 
2일 동안 발생한 총 284,807건의 거래 중 492건의 사기 거래가 포함되어 있습니다. 
데이터셋은 기밀성을 위해 PCA 변환을 통해 익명화되었으며, 'Time'과 'Amount' 특성을 제외한 모든 속성은 PCA 변환된 값으로 제공됩니다.

### Dataset Description

- **Time**: 이 거래와 데이터셋의 첫 거래 사이에 경과된 시간(초).
- **V1, V2, ..., V28**: PCA로 얻어진 주요 구성 요소. 이는 PCA 변환을 통해 익명화된 특성들입니다.
- **Amount**: 거래 금액.
- **Class**: 응답 변수로, 1은 사기 거래를 나타내고 0은 정상 거래를 나타냅니다.

### Usage

이 데이터셋은 다양한 기계 학습 및 데이터 분석 작업에 사용될 수 있습니다.

- 분류 알고리즘을 사용한 사기 탐지
- 데이터 전처리 및 특징 엔지니어링
- 모델 평가 및 성능 벤치마킹

### How to Use

1. Kaggle 페이지에서 데이터셋을 다운로드합니다: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. 다운로드한 파일의 압축을 풀고 이 폴더에 `creditcard.csv` 파일을 위치시킵니다.

### License

이 데이터셋은 Kaggle에서 공개적으로 제공되며, 학술 및 연구 목적으로 사용할 수 있습니다.

### Acknowledgements

이 데이터셋은 Université Libre de Bruxelles (ULB)의 기계 학습 그룹(MLG)에서 제공한 것입니다. 이 데이터를 공개적으로 이용할 수 있게 해준 그들의 노력에 감사드립니다.

자세한 내용과 데이터셋 다운로드는 [Kaggle 데이터셋 페이지](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)를 방문해 주세요.