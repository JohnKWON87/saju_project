"""
빠른 재학습 스크립트
모델과 전처리기를 함께 저장
"""

import os
import joblib
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer


def retrain_model():
    """모델과 전처리기를 함께 재학습"""
    
    print("=" * 50)
    print("🔄 모델 재학습 시작")
    print("=" * 50)
    
    # 1. 데이터 확인
    data_path = "./data/saju_dataset.csv"
    if not os.path.exists(data_path):
        print(f"\n❌ 데이터 파일이 없습니다: {data_path}")
        print("먼저 'python data_collector.py'를 실행하세요.")
        return False
    
    try:
        # 2. 데이터 전처리
        print("\n📊 데이터 로드 중...")
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(data_path)
        
        if df is None:
            return False
        
        print("🧹 데이터 정제 중...")
        df_clean = preprocessor.clean_data(df)
        
        print("⚙️  특성 생성 중...")
        df_features = preprocessor.create_features(df_clean)
        
        print("🔢 인코딩 중...")
        df_encoded = preprocessor.encode_categorical(
            df_features,
            columns=['계절', '주요오행']
        )
        
        print("✂️  학습/테스트 분리 중...")
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(
            df_encoded,
            target_column='성격유형'
        )
        
        # 3. 모델 학습
        print("\n🧠 모델 학습 중...")
        trainer = ModelTrainer()
        trainer.train_multiple_models(X_train, y_train)
        
        # 4. 평가
        print("\n📈 모델 평가 중...")
        accuracy = trainer.evaluate_model(X_test, y_test)
        
        # 5. 저장
        print("\n💾 저장 중...")
        os.makedirs("./models", exist_ok=True)
        
        # 모델 저장
        model_path = trainer.save_model()
        
        # ✅ 전처리기도 저장!
        preprocessor_path = "./models/preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        print(f"✅ 전처리기 저장: {preprocessor_path}")
        
        # 6. 완료
        print("\n" + "=" * 50)
        print("✅ 재학습 완료!")
        print("=" * 50)
        print(f"📊 최종 정확도: {accuracy:.2%}")
        print(f"📁 모델 파일: {model_path}")
        print(f"📁 전처리기: {preprocessor_path}")
        print("\n이제 'streamlit run app_fixed.py'로 실행하세요!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    retrain_model()