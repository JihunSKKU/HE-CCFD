## Credit Card Fraud Detection using Homomorphic Encryption (HE-CCFD)

이 프로젝트는 1D 합성곱 신경망(CNN)에 동형암호를 적용하여 암호화된 데이터의 신용카드 사기를 탐지하는 것을 목표로 합니다. 
이 프로젝트에 사용된 데이터셋은 정상 거래와 사기 거래를 포함한 익명화된 신용카드 거래 데이터를 포함하고 있습니다.

### Dataset

데이터셋은 `dataset` 폴더에 위치하며 다음 파일을 포함하고 있습니다:

- `dataset/creditcard.csv`: 모든 거래와 해당 거래의 특징이 포함된 주요 데이터셋 파일.

데이터셋에 대한 자세한 내용은 [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)를 참고해 주세요.

### Project Overview

TBD

### Homomorphic Encryption

동형암호는 암호화된 상태에서 연산을 수행할 수 있는 암호화 방식으로, 암호화된 결과를 복호화했을 때 원본 평문에서 연산한 결과와 동일한 결과를 얻을 수 있습니다. 
이를 통해 데이터의 프라이버시를 유지하면서 안전하게 데이터 처리를 할 수 있습니다.

이 프로젝트에서는 동형암호를 사용하여 모델 추론 단계에서 민감한 데이터를 보호합니다. 
이를 위해 [Lattigo](https://github.com/tuneinsight/lattigo/tree/v5.0.2) 라이브러리를 사용합니다.

### Setup Instructions

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the dataset**:
    [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)에서 데이터셋을 다운로드하여 `dataset` 디렉토리에 위치시킵니다.

### License

이 프로젝트는 MIT 라이선스에 따라 제공됩니다.

### Acknowledgements

이 데이터를 제공해주신 Université Libre de Bruxelles (ULB)의 Machine Learning Group(MLG)에 감사드립니다. 
자세한 내용과 데이터셋 다운로드는 [Kaggle 데이터셋 페이지](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)를 참조해 주세요.