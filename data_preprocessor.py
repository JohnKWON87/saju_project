"""
사주 데이터 전처리
ML 모델 학습을 위한 데이터 정제 및 변환
"""
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import MODEL_CONFIG
from config import SYSTEM_CONFIG


class DataPreprocessor:
    """사주 데이터 전처리 클래스"""
    
    def __init__(self):
        """전처리기 초기화"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, filepath="./data/saju_dataset.csv"):
        """
        CSV 파일에서 데이터 로드
        
        Args:
            filepath: CSV 파일 경로
            
        Returns:
            DataFrame: 로드된 데이터
        """
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            print(f"✓ 데이터 로드 완료: {len(df)}개 레코드")
            return df
        except FileNotFoundError:
            print(f"✗ 파일을 찾을 수 없습니다: {filepath}")
            return None
    
    def clean_data(self, df):
        """
        데이터 정제
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            DataFrame: 정제된 데이터
        """
        print("\n=== 데이터 정제 시작 ===")
        
        # 결측치 확인
        missing = df.isnull().sum()
        if missing.any():
            print(f"결측치 발견:\n{missing[missing > 0]}")
            # 결측치 제거 또는 채우기
            df = df.dropna()
        
        # 중복 제거
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"중복 제거: {duplicates}개")
            df = df.drop_duplicates()
        
        # 시간 미입력(-1)을 별도 카테고리로 처리
        df['시간입력여부'] = (df['시'] != -1).astype(int)
        
        print(f"✓ 정제 완료: {len(df)}개 레코드 남음")
        return df
    
    def create_features(self, df):
        """
        특성(Feature) 생성
        
        Args:
            df: 데이터프레임
            
        Returns:
            DataFrame: 특성이 추가된 데이터
        """
        print("\n=== 특성 생성 ===")
        
        # 1. 오행 비율 계산
        ohaeng_cols = ['목', '화', '토', '금', '수']
        df['오행합계'] = df[ohaeng_cols].sum(axis=1)
        
        for col in ohaeng_cols:
            df[f'{col}_비율'] = df[col] / df['오행합계']
        
        # 2. 오행 균형도 (표준편차)
        df['오행균형도'] = df[ohaeng_cols].std(axis=1)
        
        # 3. 계절 정보 (월 기반)
        df['계절'] = df['월'].apply(self._get_season)
        
        # 4. 연령대 (현재년도 기준)
        current_year = SYSTEM_CONFIG["CURRENT_YEAR"]
        df['연령대'] = ((current_year - df['년']) // 10) * 10
        
        print("✓ 특성 생성 완료")
        return df
    
    def _get_season(self, month):
        """월을 계절로 변환"""
        if month in [3, 4, 5]:
            return "봄"
        elif month in [6, 7, 8]:
            return "여름"
        elif month in [9, 10, 11]:
            return "가을"
        else:
            return "겨울"
    
    def encode_categorical(self, df, columns):
        """
        범주형 데이터 인코딩
        
        Args:
            df: 데이터프레임
            columns: 인코딩할 컬럼 리스트
            
        Returns:
            DataFrame: 인코딩된 데이터
        """
        print("\n=== 범주형 데이터 인코딩 ===")
        
        df_encoded = df.copy()
        
        for col in columns:
            if col in df_encoded.columns:
                # LabelEncoder 생성 및 저장
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
                print(f"✓ {col}: {len(le.classes_)} 클래스")
        
        return df_encoded
    
    def prepare_train_test(self, df, target_column='성격유형'):
        """
        학습/테스트 데이터 분리
        
        Args:
            df: 전처리된 데이터프레임
            target_column: 예측할 목표 변수
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print(f"\n=== 학습/테스트 데이터 준비 (목표: {target_column}) ===")
        
        # 특성 선택 (숫자형 특성만)
        feature_columns = [
            '목', '화', '토', '금', '수',
            '목_비율', '화_비율', '토_비율', '금_비율', '수_비율',
            '오행균형도', '시간입력여부', '연령대'
        ]
        
        # 계절 인코딩 추가
        if '계절_encoded' in df.columns:
            feature_columns.append('계절_encoded')
        
        # 사용 가능한 특성만 선택
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_names = available_features
        
        X = df[available_features]
        y = df[target_column]
        
        # 학습/테스트 분리
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=MODEL_CONFIG['TEST_SIZE'],
            random_state=MODEL_CONFIG['RANDOM_STATE'],
            stratify=y  # 클래스 비율 유지
        )
        
        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # DataFrame으로 변환 (컬럼명 유지)
        X_train = pd.DataFrame(X_train_scaled, columns=available_features)
        X_test = pd.DataFrame(X_test_scaled, columns=available_features)
        
        print(f"✓ 학습 데이터: {len(X_train)}개")
        print(f"✓ 테스트 데이터: {len(X_test)}개")
        print(f"✓ 특성 개수: {len(available_features)}")
        print(f"✓ 클래스 분포:\n{y_train.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance(self, feature_importance, top_n=10):
        """
        특성 중요도 출력
        
        Args:
            feature_importance: 모델의 특성 중요도 배열
            top_n: 상위 n개 출력
        """
        importance_df = pd.DataFrame({
            '특성': self.feature_names,
            '중요도': feature_importance
        }).sort_values('중요도', ascending=False)
        
        print(f"\n=== 특성 중요도 (Top {top_n}) ===")
        print(importance_df.head(top_n))
        
        return importance_df


# 사용 예시
if __name__ == "__main__":
    # 전처리기 생성
    preprocessor = DataPreprocessor()
    
    # 1. 데이터 로드
    df = preprocessor.load_data("./data/saju_dataset.csv")
    
    if df is not None:
        # 2. 데이터 정제
        df_clean = preprocessor.clean_data(df)
        
        # 3. 특성 생성
        df_features = preprocessor.create_features(df_clean)
        
        # 4. 범주형 데이터 인코딩
        df_encoded = preprocessor.encode_categorical(
            df_features,
            columns=['계절', '주요오행']
        )
        
        # 5. 학습/테스트 데이터 준비
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(
            df_encoded,
            target_column='성격유형'
        )
        
        print("\n✓ 전처리 완료!")
        print(f"학습 준비 완료: X_train shape = {X_train.shape}")