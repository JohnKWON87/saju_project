"""
머신러닝 모델 학습 및 평가
초급자를 위한 간단한 분류 모델
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
import os
from config import MODEL_CONFIG


class ModelTrainer:
    """머신러닝 모델 학습 클래스"""
    
    def __init__(self):
        """모델 학습기 초기화"""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_multiple_models(self, X_train, y_train):
        """
        여러 모델 학습 및 비교
        
        Args:
            X_train: 학습 데이터 특성
            y_train: 학습 데이터 레이블
        """
        print("\n=== 다양한 모델 학습 시작 ===")
        
        # 초급자용 간단한 모델들
        models = {
            "의사결정나무": DecisionTreeClassifier(
                max_depth=10,
                random_state=MODEL_CONFIG['RANDOM_STATE']
            ),
            "랜덤포레스트": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=MODEL_CONFIG['RANDOM_STATE']
            ),
            "K최근접이웃": KNeighborsClassifier(
                n_neighbors=5
            )
        }
        
        best_score = 0
        
        for name, model in models.items():
            print(f"\n--- {name} 학습 중... ---")
            
            # 모델 학습
            model.fit(X_train, y_train)
            
            # 교차 검증
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=MODEL_CONFIG['CV_FOLDS']
            )
            mean_score = cv_scores.mean()
            
            print(f"교차 검증 정확도: {mean_score:.4f} (+/- {cv_scores.std():.4f})")
            
            # 모델 저장
            self.models[name] = {
                'model': model,
                'cv_score': mean_score
            }
            
            # 최고 모델 선택
            if mean_score > best_score:
                best_score = mean_score
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n✓ 최고 성능 모델: {self.best_model_name} (정확도: {best_score:.4f})")
    
    def evaluate_model(self, X_test, y_test, model_name=None):
        """
        모델 평가
        
        Args:
            X_test: 테스트 데이터 특성
            y_test: 테스트 데이터 레이블
            model_name: 평가할 모델 이름 (None이면 최고 모델)
        """
        if model_name:
            model = self.models[model_name]['model']
        else:
            model = self.best_model
            model_name = self.best_model_name
        
        print(f"\n=== {model_name} 모델 평가 ===")
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 정확도
        accuracy = accuracy_score(y_test, y_pred)
        print(f"테스트 정확도: {accuracy:.4f}")
        
        # 분류 리포트
        print("\n상세 분류 리포트:")
        print(classification_report(y_test, y_pred))
        
        # 혼동 행렬
        print("혼동 행렬:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def get_feature_importance(self):
        """
        특성 중요도 반환 (트리 기반 모델만)
        
        Returns:
            array: 특성 중요도
        """
        if hasattr(self.best_model, 'feature_importances_'):
            return self.best_model.feature_importances_
        else:
            print("이 모델은 특성 중요도를 제공하지 않습니다.")
            return None
    
    def save_model(self, filepath=None):
        """
        학습된 모델 저장
        
        Args:
            filepath: 저장 경로 (None이면 기본 경로)
        """
        if self.best_model is None:
            print("저장할 모델이 없습니다. 먼저 학습을 진행하세요.")
            return
        
        # 디렉토리 생성
        os.makedirs("./models", exist_ok=True)
        
        if filepath is None:
            filepath = f"./models/saju_model_{self.best_model_name}.pkl"
        
        # 모델 저장
        joblib.dump(self.best_model, filepath)
        print(f"✓ 모델 저장 완료: {filepath}")
        
        return filepath
    
    def load_model(self, filepath):
        """
        저장된 모델 불러오기
        
        Args:
            filepath: 모델 파일 경로
        """
        try:
            self.best_model = joblib.load(filepath)
            print(f"✓ 모델 로드 완료: {filepath}")
            return self.best_model
        except FileNotFoundError:
            print(f"✗ 모델 파일을 찾을 수 없습니다: {filepath}")
            return None
    
    def predict(self, X):
        """
        예측 수행
        
        Args:
            X: 예측할 데이터
            
        Returns:
            array: 예측 결과
        """
        if self.best_model is None:
            print("모델이 학습되지 않았습니다.")
            return None
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """
        예측 확률 반환
        
        Args:
            X: 예측할 데이터
            
        Returns:
            array: 각 클래스별 확률
        """
        if self.best_model is None:
            print("모델이 학습되지 않았습니다.")
            return None
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            print("이 모델은 확률 예측을 지원하지 않습니다.")
            return None


# 사용 예시
if __name__ == "__main__":
    from data_preprocessor import DataPreprocessor
    
    # 1. 데이터 전처리
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data("./data/saju_dataset.csv")
    
    if df is not None:
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.create_features(df_clean)
        df_encoded = preprocessor.encode_categorical(df_features, ['계절', '주요오행'])
        
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(
            df_encoded,
            target_column='성격유형'
        )
        
        # 2. 모델 학습
        trainer = ModelTrainer()
        trainer.train_multiple_models(X_train, y_train)
        
        # 3. 모델 평가
        trainer.evaluate_model(X_test, y_test)
        
        # 4. 특성 중요도 확인
        importance = trainer.get_feature_importance()
        if importance is not None:
            preprocessor.get_feature_importance(importance)
        
        # 5. 모델 저장
        trainer.save_model()