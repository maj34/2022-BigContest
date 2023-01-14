# 2022-BigContest
다각적 모델을 활용한 대출 신청 여부 예측과 고객 군집 별 서비스 메시지 제안 : 이상치 탐지, 머신러닝, 딥러닝 모델

<br>

## 1. 배경 & 목적

- 사용자 신용 정보, 대출신청 정보, 앱 로그 정보 등의 데이터를 통한 대출 신청 예측 분석
- 군집화를 통한 고객 군 분류 및 맞춤형 서비스 메시지 제안 (평가지표: F1 Score)

<br>

## 2. 주최/주관 & 참가 대상 & 성과

- 주최: 과학기술정보통신부, NIA 한국지능정보사회진흥원
- 주관: 신한카드, LG U+, Finda, WISENUT, 제주관광공사, CJ 올리브네트웍스, KAIT 한국정보통신진흥협회
- 후원: KBD 빅데이터포럼
- 참가 대상: 전국 대학(원)생(휴학생 포함) - 전일제 대학(원)생만 해당
- 성과: **2022 빅콘테스트 데이터 분석리그 퓨처스 부문 최우수상 수상**

<br>

## 3. 대회 기간

- 제출마감: 2022년 10월 14일
- 1차 서류 심사 결과: 2022년 11월 10일
- 2차 PT 발표 심사 결과: 2022년 12월 9일

<br>

## 4. 내용

<img src='https://user-images.githubusercontent.com/75362328/212460003-16818d31-4e19-4ea4-8792-1546be2b29ce.png' width="80%" height="60%">

&nbsp;&nbsp;&nbsp;&nbsp; Finda 앱을 통해 **대출을 신청받는 고객을 예측하는 이진 분류 문제**이다. 특정 고객이 대출 신청을 하기에는 한 가지 요소로 결정되지 않으므로 데이터 탐색 및 시각화를 통해 선정된 큰 관점인 **‘고객 상황’, ‘고객 행동’, ‘대출 상품’** 에 따라 분석을 진행하였다.

&nbsp;&nbsp;&nbsp;&nbsp; 또한 외부 데이터를 수집하는 과정에서도 2가지 관점을 가지고 수집하였다. 코인 가격, 금리 정보, 실업자 및 실업률, Kospi 지수 등 ‘**외부 경제 상황에 대한 정보**’와 소비자 심리 지수와 같은 ‘**소비자 내부의 결정 요인**’이 그 예이다.

&nbsp;&nbsp;&nbsp;&nbsp; 이상치 처리 및 결측치 처리에서는 **KNN Imputer**나 **RandomForest** 같은 모델을 사용하여 기계적으로 처리하기도 했지만 주어진 데이터의 특성을 최대한 반영하기 위해서 **정성적인 방법을 많이 사용**해주었다. 피처 엔지니어링 과정에서도 **Application 별/User 별/시간 별**로 구분이 가능한 피처로 나누어 총 71가지 피처를 만들어냈다. 

&nbsp;&nbsp;&nbsp;&nbsp; 모델링은 **Pycaret**이라는 AutoML 라이브러리를 사용하여 다양한 모델을 실험해 보았고 그 결과 전통적인 머신러닝 모델 중 Boosting 계열 모델이 성능이 높음을 확인할 수 있었다. 데이터가 95대 5로 매우 불균형 데이터임에 따라 다양한 이상치 탐지 모델링과 Undersampling 및 Oversampling 과정을 실험해 보았다. 하지만 결론적으로는 데이터의 분포를 건드리는 것보다 모델 자체적으로 weight를 주어 target 값의 비율을 조정하는 방법이 오히려 성능을 좋은 것을 확인하였다. 따라서 **LGBM, Catboost 모델의 scale_pos_weight라는 하이퍼파라미터를 조정하여 불균형 데이터를 샘플링** 해주었다. 마지막으로 TabNet이라는 딥러닝 계열의 모델을 사용해 최종적으로 **LGBM, Catboost, TabNet에서 학습된 결과를 취합하여 hard voting 방식의 앙상블**을 해주었다.

&nbsp;&nbsp;&nbsp;&nbsp; 또한 고객 별 **군집화를 수행하고 서비스 메시지를 제안**하는 것이 2번째 과제였는데 고객 스펙과 관련된 정보의 수치형 변수들을 사용하여 **K-means Clustering을 진행**하였다. 그에 따라서 총 5개의 군집을 얻을 수 있었는데, 나누어진 군집에 따라 특징이 매우 달랐다. 이에 따라 고객을 3개의 층위로 구분하여 각각의 층위에 맞는 메시지를 제안하고자 했다. 예를 들어 **신규 고객**의 경우 핀다의 다양한 기능들을 최대한 경험할 수 있도록 핀다의 주요 서비스를 소개할 수 있는 튜토리얼을 제공하는 아이디어를 제안했고, 앱을 사용한 경험이 있는 고객의 경우 앞서 수행한 클러스터링 결과를 활용하여 클러스터별 행동 경향에 따라서 **활성 고객/비활성 고객** 별로 맞춤화된 서비스 메시지를 제안했다. 특히 최대한 고객들이 사용해 보지 않은 서비스를 추천하도록 했다.

<br>

## 5. 담당 역할

- SMOTE-TOMEK 방법을 사용한 불균형 데이터 샘플링
- Pycaret을 사용한 AutoML / Weight를 조정한 LGBM, Catboost 모델 학습 및 최적화
- K-Means Clustering, Self-Organizing Map 등을 활용한 군집화
- Feature Importance, SHAP 등의 메소드를 활용한 모델 설명력 평가

<br>

## 6. Process

---

### ch.1 EDA

- Target 값 분석
- 고객 상황 별 분석
- 고객 행동 별 분석
- 대출 상품 별 분석

---

### ch.2 Data Preparation

- 외부 데이터 수집
    - 외부 경제 상황에 대한 정보
    - 소비자 내부의 결정 요인

---

### ch.3 Preprocessing

- 이상치 처리
- 결측치 처리
- 피처 엔지니어링
    - Application 별
    - User 별
    - 시간 별

---

### ch.4 Modeling

- 데이터 검증 방법 설정
- 불균형 데이터 샘플링
- 이상치 탐지 모델링
    - Isolation Forest
    - K-Nearest Neighbor
    - Minimum Covariance Determinant
- 머신러닝 모델링
    - Pycaret
    - LGBM
    - Catboost
- 딥러닝 모델링
    - TabNet

---

### ch.5 Ensemble

- Hard Voting
    - LGBM
    - Catboost
    - TabNet

---

### ch.6 Clustering & Service Message

- K-means Clustering

<br>

## 7. 참고자료

[2022 빅콘테스트 발표 자료](https://drive.google.com/file/d/1Hz6fGXc3Ie0JZlEk4dD6MuGHR_TLl2kp/view?usp=share_link)

[유튜브 발표 영상](https://www.youtube.com/watch?v=0X0B3eQtRZA)

<br>

## 8. 증빙자료

[빅콘테스트 사이트](https://www.bigcontest.or.kr/community/view.php?No=648&gubun=notice&keyfield=&keyword=&page=1&keyCate=)

<img src='https://user-images.githubusercontent.com/75362328/212461129-41794954-ec06-4908-9bf9-ae947f7fba6b.jpg' width='80%' height='60%'>
<img src='https://user-images.githubusercontent.com/75362328/212461682-4bf57078-37be-47a0-a133-b31df1b92b36.jpg' width='60%' height='40%'>
<img src='https://user-images.githubusercontent.com/75362328/212461135-22275f73-e31a-44ef-81a8-2c80113c7f53.jpg' width='60%' height='40%'>
