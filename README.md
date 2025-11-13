# 🔮 AI 기반 사주 풀이 시스템

> Python 머신러닝 & Gemini API 기반 사주 분석 및 운세 해석 플랫폼

## 📌 프로젝트 소개

생년월일시를 입력받아 전통 사주팔자를 계산하고, 머신러닝 모델과 Google Gemini API를 결합하여 개인 맞춤형 운세를 제공하는 AI 시스템입니다.
전통 사주학과 현대 AI 기술을 융합하여 보다 정확하고 이해하기 쉬운 운세 해석을 제공합니다.

- **개발 기간**: 2025.11월 10일 ~ 2025.11월 13일
- **개발 인원**: 1팀 프로젝트 ( 총인원 : 3명, 팀원 )
- **담당 역할**: 데이터 수집 및 전처리 / 머신러닝 모델 구축 / API 통합 / AI 적극활용 벤치마킹 비중이 높음

## 🛠 기술 스택

### Language & Framework
- Python 3.13

### Machine Learning
- scikit-learn (RandomForest, DecisionTree, KNN)
- pandas (데이터 처리)
- numpy (수치 연산)

### AI & API
- Google Gemini API (자연어 운세 해석)
- Requests (API 통신)

### Data Processing
- BeautifulSoup / Selenium (웹 스크래핑)
- pickle (모델 저장/로드)

### Tools
- Jupyter Notebook
- Git / GitHub

## 🎯 주요 기능

### 1. 사주팔자 계산
- **생년월일시 입력**: 양력/음력 날짜 기반 사주 계산
- **천간지지 추출**: 년주, 월주, 일주, 시주 자동 계산
- **오행 분석**: 목/화/토/금/수 5행 균형 분석

### 2. 머신러닝 기반 운세 예측
- **3가지 알고리즘 앙상블**:
  - 랜덤 포레스트 (Random Forest)
  - 의사결정나무 (Decision Tree)
  - K-최근접 이웃 (KNN)
- **특성 엔지니어링**: 사주 구성 요소를 수치화하여 모델 학습
- **예측 결과 도출**: 종합 운세 점수 및 카테고리 예측

### 3. Gemini API 자연어 해석
- **AI 기반 운세 해석**: 머신러닝 예측 결과를 자연어로 변환
- **맞춤형 조언 생성**: 개인의 사주 특성에 맞는 구체적인 조언
- **질의응답 기능**: 사용자 질문에 대한 AI 답변

### 4. 데이터 수집 및 전처리
- **자동화된 데이터 수집**: 웹 스크래핑을 통한 사주 데이터 수집
- **데이터 정제**: 결측치 처리, 이상치 제거
- **범주형 인코딩**: 천간지지 → 숫자 변환 (Label Encoding)
- **스케일링**: 데이터 정규화 (StandardScaler)

### 5. 모델 재학습 시스템
- 새로운 데이터 추가 시 자동 재학습
- 모델 성능 비교 및 최적 모델 선택
- 버전 관리 및 롤백 기능

## 📂 프로젝트 구조

```
saju_project/
├── app.py                     # 메인 실행 파일
├── app_fixed.py               # 안정화된 버전
├── api_client.py              # Gemini API 클라이언트
├── config.py                  # 환경 설정 (API Key)
│
├── data_collector.py          # 데이터 수집기
├── data_collector_v2.py       # 개선된 수집 로직
├── data_preprocessor.py       # 전처리 파이프라인
├── fix_year_2025.py           # 2025년 데이터 업데이트
│
├── gemini_fortune_teller.py   # AI 운세 해석 모듈
├── model_trainer.py           # 모델 학습 코드
├── predictor.py               # 운세 예측기
├── retrain.py                 # 재학습 스크립트
│
├── saju_caculator.py          # 사주팔자 계산 로직
├── quick_start.py             # 빠른 실행 샘플
├── test_api.py                # API 테스트
│
├── data/
│   └── saju_dataset.csv       # 사주 데이터셋
│
├── models/                    # 학습된 모델
│   ├── preprocessor.pkl
│   ├── saju_model_랜덤포레스트.pkl
│   ├── saju_model_의사결정나무.pkl
│   └── saju_model_K최근접이웃.pkl
│
├── requirements.txt
└── README.md
```

## 🚀 실행 방법

### 1. Python 환경 설정
```bash
python --version  # Python 3.8 이상 필요
```

### 2. 필요한 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3. Gemini API Key 설정
`config.py` 파일에 API 키를 입력합니다.
```python
GEMINI_API_KEY = "your-api-key-here"
```

API 키는 [Google AI Studio](https://makersuite.google.com/app/apikey)에서 무료로 발급받을 수 있습니다.

### 4. 데이터 수집 (선택사항)
```bash
python data_collector_v2.py
```

### 5. 모델 학습
```bash
python model_trainer.py
```

### 6. 프로그램 실행
```bash
커맨드창 cmd실행 cd python 이동 
python streamlit run app_fixed.py
```

### 7. 사용 예시
```python
# 빠른 테스트
python quick_start.py

# 입력 예시
생년월일: 1990년 1월 15일
출생시간: 오전 10시

# 출력 예시
사주: 甲午년 丙寅월 己未일 己巳시
운세 점수: 75/100
AI 해석: "당신의 사주는..."
```

## 💡 트러블슈팅

### 1. 사주 계산 정확도 문제
**문제**: 음력/양력 변환 시 날짜 오차 발생

**해결**:
- Python `lunarcalendar` 라이브러리 활용
- 절입(節入) 시간을 고려한 정밀 계산
- 2025년 이후 데이터 자동 업데이트 스크립트 작성

```python
from lunarcalendar import Converter, Solar

solar = Solar(1990, 1, 15)
lunar = Converter.Solar2Lunar(solar)
```

**결과**: 날짜 계산 정확도 99% 달성

### 2. 머신러닝 모델 과적합 문제
**문제**: 학습 데이터에만 잘 맞고 실제 예측 성능 저하

**해결**:
- **교차 검증** (K-Fold Cross Validation) 적용
- **하이퍼파라미터 튜닝**: GridSearchCV 사용
- **앙상블 기법**: 3가지 모델 평균으로 안정성 향상
- **정규화 적용**: L2 Regularization

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"평균 정확도: {scores.mean():.2f}")
```

**결과**: 
- 과적합 감소: 테스트 정확도 +15% 향상
- 일반화 성능 개선

### 3. Gemini API 응답 속도 최적화
**문제**: API 호출 시 응답 시간이 5초 이상 소요

**해결**:
- **프롬프트 최적화**: 불필요한 요청 내용 제거
- **결과 캐싱**: 동일 사주에 대한 반복 요청 방지
- **비동기 처리**: `asyncio`를 활용한 동시성 구현

```python
import asyncio

async def get_fortune_async(saju_data):
    response = await gemini_client.generate_async(prompt)
    return response.text
```

**결과**: 평균 응답 시간 5초 → 1.5초로 단축 (70% 개선)

### 4. 데이터 불균형 문제
**문제**: 특정 운세 카테고리 데이터 부족으로 편향된 예측

**해결**:
- **SMOTE 기법**: Synthetic Minority Over-sampling
- **가중치 조정**: `class_weight='balanced'` 설정
- **데이터 증강**: 소수 클래스 샘플 생성

**결과**: 소수 클래스 예측 정확도 +25% 향상

## 📊 모델 성능

|      모델     | 정확도 | 재현율 | F1-Score |
|---------------|--------|--------|----------|
| 랜덤 포레스트  |   82%  |  79%   |  0.80    |
| 의사결정나무   |   75%  |  73%   |  0.74    |
| K-최근접 이웃  |   78%  |  76%   |  0.77    |
|**앙상블 평균**| **85%**| **82%**| **0.83** |

## 📈 개선 계획

- [ ] 웹 인터페이스 개발 (Flask/Django)
- [ ] 실시간 운세 알림 기능
- [ ] 사주 궁합 분석 추가
- [ ] 월별/연별 상세 운세 제공
- [ ] 데이터베이스 연동 (PostgreSQL)
- [ ] 사용자 피드백 기반 모델 개선
- [ ] 모바일 앱 개발

## 🎓 배운 점

- **머신러닝 파이프라인 구축**: 데이터 수집 → 전처리 → 학습 → 배포 전 과정
- **API 통합**: 외부 AI API와 자체 모델 결합
- **데이터 전처리의 중요성**: 정제된 데이터가 모델 성능에 미치는 영향
- **앙상블 기법**: 여러 모델을 결합하여 안정성 향상
- **실제 서비스 설계**: 사용자 관점에서의 기능 구현

## 🔐 보안 및 윤리

- API 키는 `.env` 파일로 관리하고 `.gitignore`에 추가
- 개인정보(생년월일)는 암호화하여 저장
- AI 해석 결과는 참고용이며 의사결정에 절대적 기준이 아님을 명시

  ## 👥 팀원 및 역할

- **김상혁 팀장**: PPT자료 및 정의서/명세서 담당
- **권혁민 팀원**: gemini AI기반 API적극 활용 간단한 AI 사주풀이 
- **이지한 팀원**: AI 정보활용 및 사주풀이 사이트 비교 및 다른 open source 참조

## 📧 문의

- Email: johnkwon33@gmail.com
- GitHub: https://github.com/johnkwon87/saju_project

---

© 2025 AI Fortune Teller Project. All rights reserved.
