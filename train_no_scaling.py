"""
스케일링 없이 모델 학습
StandardScaler 오류 완전 해결
"""
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os


def load_and_prepare_data(filepath="./data/saju_dataset.csv"):
    """데이터 로드 및 전처리 (스케일링 없음)"""
    
    print("\n=== 데이터 로드 ===")
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    print(f"✅ {len(df)}개 레코드 로드")
    
    # 결측치 제거
    df = df.dropna()
    
    # 시간 입력 여부
    df['시간입력여부'] = (df['시'] != -1).astype(int)
    
    # 오행 비율 계산
    ohaeng_cols = ['목', '화', '토', '금', '수']
    df['오행합계'] = df[ohaeng_cols].sum(axis=1)
    
    for col in ohaeng_cols:
        df[f'{col}_비율'] = df[col] / df['오행합계']
    
    # 오행 균형도
    df['오행균형도'] = df[ohaeng_cols].std(axis=1)
    
    # 계절
    def get_season(month):
        if month in [3, 4, 5]:
            return 0  # 봄
        elif month in [6, 7, 8]:
            return 1  # 여름
        elif month in [9, 10, 11]:
            return 2  # 가을
        else:
            return 3  # 겨울
    
    df['계절_encoded'] = df['월'].apply(get_season)
    
    # 연령대
    df['연령대'] = ((datetime.now().year - df['년']) // 10) * 10
    
    print("✅ 전처리 완료")
    
    return df


def train_model_no_scaling(df, target_column='성격유형'):
    """스케일링 없이 모델 학습"""
    
    print(f"\n=== 모델 학습 (목표: {target_column}) ===")
    
    # 특성 선택
    feature_columns = [
        '목', '화', '토', '금', '수',
        '목_비율', '화_비율', '토_비율', '금_비율', '수_비율',
        '오행균형도', '시간입력여부', '연령대', '계절_encoded'
    ]
    
    X = df[feature_columns]
    y = df[target_column]
    
    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"✅ 학습: {len(X_train)}개, 테스트: {len(X_test)}개")
    
    # 여러 모델 학습
    models = {
        "의사결정나무": DecisionTreeClassifier(max_depth=10, random_state=42),
        "랜덤포레스트": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n--- {name} 학습 중 ---")
        
        # 학습
        model.fit(X_train, y_train)
        
        # 교차 검증
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_score = cv_scores.mean()
        
        print(f"교차 검증 정확도: {mean_score:.4f} (+/- {cv_scores.std():.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name
    
    # 테스트 평가
    print(f"\n=== 최고 모델: {best_name} ===")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"테스트 정확도: {accuracy:.4f}")
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred))
    
    return best_model, best_name, accuracy


def save_model(model, model_name):
    """모델 저장"""
    os.makedirs("./models", exist_ok=True)
    filepath = f"./models/saju_model_{model_name}.pkl"
    joblib.dump(model, filepath)
    print(f"\n✅ 모델 저장: {filepath}")
    return filepath


def main():
    """메인 실행"""
    
    print("=" * 60)
    print("🧠 스케일링 없는 사주 모델 학습")
    print("=" * 60)
    
    # 1. 데이터 확인
    data_path = "./data/saju_dataset.csv"
    if not os.path.exists(data_path):
        print(f"\n❌ 데이터 파일 없음: {data_path}")
        print("먼저 'python data_collector.py'를 실행하세요.")
        return
    
    try:
        # 2. 데이터 로드 및 전처리
        df = load_and_prepare_data(data_path)
        
        # 3. 모델 학습
        model, model_name, accuracy = train_model_no_scaling(df, target_column='성격유형')
        
        # 4. 저장
        model_path = save_model(model, model_name)
        
        # 5. 완료
        print("\n" + "=" * 60)
        print("✅ 학습 완료!")
        print("=" * 60)
        print(f"📊 최종 정확도: {accuracy:.2%}")
        print(f"📁 모델 파일: {model_path}")
        print("\n다음 명령어로 테스트하세요:")
        print("  python simple_predictor.py")
        print("\n또는 웹앱 실행:")
        print("  streamlit run app_fixed.py")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()