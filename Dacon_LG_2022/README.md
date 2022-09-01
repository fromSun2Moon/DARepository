# Dacon 

---------

- This repository contains source code to dacon project for regression task (multi-output). 
- 비슷한 태스크(Regression)에 사용할수 있는 machine learning/deep learning 코드 베이스라인입니다.  
- 예제 대회 : LG 자율주행 센서 안테나 성능 예측.

* [자율주행]](https://arxiv.org/pdf/2101.06804)
---

## 1. Data

Download and unzip [open.zip](https://dacon.io/competitions/official/235927/data) and move data directory. 


## 2. Baselines

- 데이터 피처링보다는 모델 탐색위주로 실험. 
  - 1) 전통적인 머신러닝 방법론들 (e.g. 부스팅 알고리즘, 트리 알고리즘 등).
    - 1-1) 알고리즘을 섞어서 (voting) 가장 좋은 조합을 찾아내서 모델과 그 기록을 저장하는 코드. 
  - 2) 딥러닝 계열 multi-perceptron layers 변형 방법론을 이용하여 실험.
    - 2-1) 기본적인 멀티 퍼셉트론 + Short-skip connection을 적용하여 가장 좋은 기록의 모델과 그 기록을 저장하는 코드.

### How to Run?

```
python ml_regressors.py
python dl_regressors.py

```
 - 필요한 모듈들은 utils, model 파일들로 정리.  

## 3. Save and Test

- test.csv 및 sample_submission.csv 파일을 읽어서 베스트 스코어(e.g. valid loss, rmse score ) 저장할 경로 디렉토리 및 파일 이름 지정 후 저장. 


