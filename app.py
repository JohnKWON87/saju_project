"""
사주풀이 웹 애플리케이션
Streamlit 기반 인터페이스
"""

import streamlit as st
import pandas as pd
from api_client import SajuAPIClient
from model_trainer import ModelTrainer
from data_preprocessor import DataPreprocessor
from predictor import SajuPredictor
import os


# 페이지 설정
st.set_page_config(
    page_title="AI 사주풀이",
    page_icon="🔮",
    layout="wide"
)


def main():
    """메인 애플리케이션"""
    
    st.title("🔮 AI 사주풀이 시스템")
    st.markdown("---")
    
    # 사이드바 메뉴
    menu = st.sidebar.selectbox(
        "메뉴 선택",
        ["사주 보기", "데이터 수집", "모델 학습", "프로젝트 정보"]
    )
    
    if menu == "사주 보기":
        show_saju_prediction()
    elif menu == "데이터 수집":
        show_data_collection()
    elif menu == "모델 학습":
        show_model_training()
    else:
        show_project_info()


def show_saju_prediction():
    """사주 예측 페이지"""
    
    st.header("📅 사주 보기")
    
    # 입력 폼
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name = st.text_input("이름 (선택)", placeholder="홍길동")
        birth_year = st.number_input("출생 연도", min_value=1900, max_value=2024, value=1990)
    
    with col2:
        birth_month = st.number_input("출생 월", min_value=1, max_value=12, value=5)
        birth_day = st.number_input("출생 일", min_value=1, max_value=31, value=15)
    
    with col3:
        use_hour = st.checkbox("출생 시간 입력")
        if use_hour:
            birth_hour = st.number_input("출생 시간", min_value=0, max_value=23, value=12)
        else:
            birth_hour = None
    
    if st.button("사주 풀이 시작", type="primary"):
        with st.spinner("사주를 분석 중입니다..."):
            try:
                # 모델 로드
                preprocessor = DataPreprocessor()
                trainer = ModelTrainer()
                
                model_path = "./models/saju_model_랜덤포레스트.pkl"
                
                if os.path.exists(model_path):
                    model = trainer.load_model(model_path)
                    
                    # API 클라이언트 및 예측기 생성
                    api_client = SajuAPIClient()
                    predictor = SajuPredictor(model, preprocessor, api_client)
                    
                    # 사주 예측
                    result = predictor.predict_saju(
                        birth_year=birth_year,
                        birth_month=birth_month,
                        birth_day=birth_day,
                        birth_hour=birth_hour,
                        name=name if name else None
                    )
                    
                    # 결과 표시
                    display_saju_result(result)
                    
                else:
                    st.error("모델이 학습되지 않았습니다. '모델 학습' 메뉴에서 먼저 학습을 진행하세요.")
                    
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")


def display_saju_result(result):
    """사주 결과 표시"""
    
    st.success("✅ 사주 풀이 완료!")
    
    # 기본 정보
    st.subheader("📋 기본 정보")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**입력 정보**\n\n{result['입력정보']['양력']}")
    
    with col2:
        st.info(f"**사주팔자**\n\n"
                f"년주: {result['사주팔자']['년주']}\n\n"
                f"월주: {result['사주팔자']['월주']}\n\n"
                f"일주: {result['사주팔자']['일주']}\n\n"
                f"시주: {result['사주팔자']['시주']}")
    
    # 오행 분석
    st.subheader("🌟 오행 분석")
    ohaeng_df = pd.DataFrame([result['오행분석']])
    st.bar_chart(ohaeng_df.T)
    
    # AI 예측
    st.subheader("🤖 AI 분석")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("성격 유형", result['AI예측']['성격유형'])
    
    with col2:
        st.metric("신뢰도", result['AI예측']['신뢰도'])
    
    # 해석
    st.subheader("📖 해석")
    
    st.markdown(f"**전체운**\n\n{result['해석']['전체운']}")
    st.markdown(f"**성격**\n\n{result['해석']['성격']}")
    st.markdown(f"**조언**\n\n{result['해석']['조언']}")


def show_data_collection():
    """데이터 수집 페이지"""
    
    st.header("📊 데이터 수집")
    
    st.info("API를 사용해 학습용 데이터를 수집합니다.")
    
    num_samples = st.number_input(
        "수집할 샘플 수",
        min_value=10,
        max_value=1000,
        value=50,
        step=10
    )
    
    if st.button("데이터 수집 시작"):
        with st.spinner(f"{num_samples}개 샘플 수집 중..."):
            try:
                from data_collector import DataCollector
                
                api_client = SajuAPIClient()
                collector = DataCollector(api_client)
                
                # 데이터 수집
                collector.collect_sample_data(num_samples=num_samples)
                
                # 저장
                df = collector.save_to_csv("saju_dataset.csv")
                
                st.success(f"✅ {len(df)}개 샘플 수집 완료!")
                
                # 미리보기
                st.subheader("데이터 미리보기")
                st.dataframe(df.head(10))
                
                # 통계
                st.subheader("데이터 통계")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**오행 분포**")
                    st.bar_chart(df['주요오행'].value_counts())
                
                with col2:
                    st.write("**성격 유형 분포**")
                    st.bar_chart(df['성격유형'].value_counts())
                
            except Exception as e:
                st.error(f"데이터 수집 중 오류: {e}")


def show_model_training():
    """모델 학습 페이지"""
    
    st.header("🧠 모델 학습")
    
    st.info("수집된 데이터로 AI 모델을 학습합니다.")
    
    # 데이터 확인
    if os.path.exists("./data/saju_dataset.csv"):
        df = pd.read_csv("./data/saju_dataset.csv")
        st.success(f"✅ 데이터셋 로드 완료 ({len(df)}개 샘플)")
        
        target = st.selectbox(
            "예측할 목표 변수 선택",
            ["성격유형", "운세유형", "주요오행"]
        )
        
        if st.button("모델 학습 시작", type="primary"):
            with st.spinner("모델 학습 중... (시간이 걸릴 수 있습니다)"):
                try:
                    # 전처리
                    preprocessor = DataPreprocessor()
                    df_clean = preprocessor.clean_data(df)
                    df_features = preprocessor.create_features(df_clean)
                    df_encoded = preprocessor.encode_categorical(
                        df_features,
                        ['계절', '주요오행']
                    )
                    
                    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(
                        df_encoded,
                        target_column=target
                    )
                    
                    # 모델 학습
                    trainer = ModelTrainer()
                    trainer.train_multiple_models(X_train, y_train)
                    
                    # 평가
                    accuracy = trainer.evaluate_model(X_test, y_test)
                    
                    # 모델 저장
                    trainer.save_model()
                    
                    st.success(f"✅ 모델 학습 완료! (정확도: {accuracy:.2%})")
                    
                    # 특성 중요도
                    importance = trainer.get_feature_importance()
                    if importance is not None:
                        st.subheader("특성 중요도")
                        importance_df = preprocessor.get_feature_importance(importance)
                        st.bar_chart(importance_df.set_index('특성')['중요도'].head(10))
                    
                except Exception as e:
                    st.error(f"학습 중 오류: {e}")
    else:
        st.warning("⚠️ 데이터셋이 없습니다. '데이터 수집' 메뉴에서 먼저 데이터를 수집하세요.")


def show_project_info():
    """프로젝트 정보 페이지"""
    
    st.header("ℹ️ 프로젝트 정보")
    
    st.markdown("""
    ### 🔮 AI 사주풀이 시스템
    
    **프로젝트 개요**
    - 한국천문연구원 음양력 API 활용
    - 머신러닝 기반 성격/운세 예측
    - Streamlit 웹 인터페이스
    
    **기술 스택**
    - Python 3.x
    - scikit-learn (머신러닝)
    - pandas (데이터 처리)
    - Streamlit (웹 인터페이스)
    - 공공데이터 API
    
    **주요 기능**
    1. 생년월일 기반 사주팔자 계산
    2. 오행 분석
    3. AI 기반 성격/운세 예측
    4. 데이터 수집 및 모델 학습
    
    **사용 방법**
    1. **데이터 수집**: API로 학습 데이터 수집
    2. **모델 학습**: 수집한 데이터로 AI 모델 학습
    3. **사주 보기**: 생년월일 입력 후 사주 확인
    
    **참고사항**
    - 공공데이터포털에서 API 키 발급 필요
    - `config.py`에 API 키 입력 필수
    
    **API 신청 링크**
    - [한국천문연구원 음양력 API](https://www.data.go.kr/data/15012679/openapi.do)
    """)


if __name__ == "__main__":
    main()