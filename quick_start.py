"""
사주풀이 프로젝트 빠른 시작 스크립트
API 키 없이도 실행 가능
"""

import os
import sys

def check_dependencies():
    """필요한 라이브러리 확인"""
    print("=== 라이브러리 확인 중 ===")
    required = ['pandas', 'numpy', 'sklearn', 'streamlit']
    missing = []
    
    for lib in required:
        try:
            __import__(lib)
            print(f"✓ {lib}")
        except ImportError:
            print(f"✗ {lib} - 설치 필요")
            missing.append(lib)
    
    if missing:
        print(f"\n⚠️  다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("✓ 모든 라이브러리 설치됨\n")
    return True


def setup_directories():
    """필요한 디렉토리 생성"""
    print("=== 디렉토리 설정 ===")
    dirs = ['./data', './models', './output']
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ {d}")
    print()


def collect_data():
    """데이터 수집 (API 불필요)"""
    print("=== 데이터 수집 시작 ===")
    
    try:
        from data_collector import DataCollectorV2
        
        collector = DataCollectorV2()
        collector.collect_sample_data(num_samples=100)
        df = collector.save_to_csv("saju_dataset.csv")
        
        print(f"✓ 데이터 수집 완료: {len(df)}개\n")
        return True
    except Exception as e:
        print(f"✗ 데이터 수집 실패: {e}\n")
        return False


def train_model():
    """모델 학습"""
    print("=== 모델 학습 시작 ===")
    
    try:
        from data_preprocessor import DataPreprocessor
        from model_trainer import ModelTrainer
        
        # 데이터 전처리
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data("./data/saju_dataset.csv")
        
        if df is None:
            print("✗ 데이터 파일 없음")
            return False
        
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.create_features(df_clean)
        df_encoded = preprocessor.encode_categorical(
            df_features,
            columns=['계절', '주요오행']
        )
        
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(
            df_encoded,
            target_column='성격유형'
        )
        
        # 모델 학습
        trainer = ModelTrainer()
        trainer.train_multiple_models(X_train, y_train)
        
        # 평가
        accuracy = trainer.evaluate_model(X_test, y_test)
        
        # 저장
        trainer.save_model()
        
        print(f"✓ 모델 학습 완료 (정확도: {accuracy:.2%})\n")
        return True
        
    except Exception as e:
        print(f"✗ 모델 학습 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_app():
    """웹앱 실행"""
    print("=== Streamlit 앱 실행 ===")
    print("브라우저가 자동으로 열립니다...")
    print("종료하려면 Ctrl+C를 누르세요\n")
    
    os.system("streamlit run app.py")


def main():
    """메인 실행 함수"""
    print("╔════════════════════════════════════╗")
    print("║   🔮 AI 사주풀이 빠른 시작      ║")
    print("╚════════════════════════════════════╝\n")
    
    # 1. 라이브러리 확인
    if not check_dependencies():
        print("\n❌ 라이브러리를 먼저 설치해주세요")
        sys.exit(1)
    
    # 2. 디렉토리 설정
    setup_directories()
    
    # 3. 데이터 확인/수집
    if not os.path.exists("./data/saju_dataset.csv"):
        print("📊 데이터 파일이 없습니다. 수집을 시작합니다...\n")
        if not collect_data():
            print("\n❌ 데이터 수집 실패")
            sys.exit(1)
    else:
        print("✓ 데이터 파일 존재\n")
    
    # 4. 모델 확인/학습
    if not os.path.exists("./models/saju_model_랜덤포레스트.pkl"):
        print("🧠 모델 파일이 없습니다. 학습을 시작합니다...\n")
        if not train_model():
            print("\n❌ 모델 학습 실패")
            sys.exit(1)
    else:
        print("✓ 모델 파일 존재\n")
    
    # 5. 웹앱 실행
    print("✅ 모든 준비 완료!\n")
    run_app()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()