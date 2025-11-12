"""
사주 예측 및 해석 생성
학습된 모델로 운세 예측
"""
from datetime import datetime

import pandas as pd
import numpy as np
from api_client import SajuAPIClient
from config import FORTUNE_TEMPLATES
import random
from config import SYSTEM_CONFIG


class SajuPredictor:
    """사주 예측 및 해석 생성 클래스"""
    
    def __init__(self, model, preprocessor, api_client):
        """
        예측기 초기화
        
        Args:
            model: 학습된 ML 모델
            preprocessor: 데이터 전처리기
            api_client: API 클라이언트
        """
        self.model = model
        self.preprocessor = preprocessor
        self.api_client = api_client
    
    def predict_saju(self, birth_year, birth_month, birth_day, 
                     birth_hour=None, name=None):
        """
        생년월일로 사주 예측
        
        Args:
            birth_year: 출생 연도
            birth_month: 출생 월
            birth_day: 출생 일
            birth_hour: 출생 시간 (선택)
            name: 이름 (선택)
            
        Returns:
            dict: 사주 예측 결과
        """
        print(f"\n=== 사주 풀이 시작 ===")
        print(f"입력: {birth_year}년 {birth_month}월 {birth_day}일", end="")
        if birth_hour is not None:
            print(f" {birth_hour}시")
        else:
            print(" (시간 미입력)")
        
        # 1. API로 사주 정보 가져오기
        saju_data = self.api_client.calculate_saju(
            birth_year=birth_year,
            birth_month=birth_month,
            birth_day=birth_day,
            birth_hour=birth_hour
        )
        
        if not saju_data:
            return {"error": "사주 정보를 가져올 수 없습니다."}
        
        # 2. 특성 생성
        features = self._create_features_from_saju(
            saju_data,
            birth_year,
            birth_month,
            birth_hour
        )
        
        # 3. ML 모델로 예측
        prediction = self.model.predict(features)[0]
        
        # 확률이 있으면 가져오기
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
        else:
            confidence = 1.0
        
        # 4. 해석 생성
        interpretation = self._generate_interpretation(
            saju_data,
            prediction,
            confidence,
            name
        )
        
        # 5. 결과 조합
        result = {
            "입력정보": saju_data['입력정보'],
            "사주팔자": saju_data['사주팔자'],
            "오행분석": saju_data['오행분석'],
            "AI예측": {
                "성격유형": prediction,
                "신뢰도": f"{confidence*100:.1f}%"
            },
            "해석": interpretation
        }
        
        return result
    
    def _create_features_from_saju(self, saju_data, birth_year, 
                                    birth_month, birth_hour):
        """
        사주 데이터에서 ML 특성 생성
        
        Args:
            saju_data: API로 받은 사주 데이터
            birth_year: 출생 연도
            birth_month: 출생 월
            birth_hour: 출생 시간
            
        Returns:
            DataFrame: 모델 입력용 특성
        """
        ohaeng = saju_data['오행분석']
        
        # 오행 합계
        total = sum(ohaeng.values())
        
        # 특성 딕셔너리 생성
        features = {
            '목': ohaeng['목'],
            '화': ohaeng['화'],
            '토': ohaeng['토'],
            '금': ohaeng['금'],
            '수': ohaeng['수'],
            '목_비율': ohaeng['목'] / total if total > 0 else 0,
            '화_비율': ohaeng['화'] / total if total > 0 else 0,
            '토_비율': ohaeng['토'] / total if total > 0 else 0,
            '금_비율': ohaeng['금'] / total if total > 0 else 0,
            '수_비율': ohaeng['수'] / total if total > 0 else 0,
            '오행균형도': np.std(list(ohaeng.values())),
            '시간입력여부': 1 if birth_hour is not None else 0,
            '연령대': ((SYSTEM_CONFIG["CURRENT_YEAR"] - birth_year) // 10) * 10
        }
        
        # 계절 인코딩
        season = self._get_season(birth_month)
        season_map = {"봄": 0, "여름": 1, "가을": 2, "겨울": 3}
        features['계절_encoded'] = season_map.get(season, 0)
        
        # DataFrame 생성 및 스케일링
        df = pd.DataFrame([features])
        
        # 전처리기의 스케일러 사용
        if hasattr(self.preprocessor, 'scaler'):
            df_scaled = self.preprocessor.scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns)
        
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
    
    def _generate_interpretation(self, saju_data, prediction, 
                                  confidence, name):
        """
        사주 해석 생성
        
        Args:
            saju_data: 사주 데이터
            prediction: AI 예측 결과
            confidence: 예측 신뢰도
            name: 이름
            
        Returns:
            dict: 해석 텍스트
        """
        ohaeng = saju_data['오행분석']
        
        # 주요 오행 찾기
        main_ohaeng = max(ohaeng, key=ohaeng.get)
        weak_ohaeng = min(ohaeng, key=ohaeng.get)
        
        # 템플릿에서 정보 가져오기
        main_template = FORTUNE_TEMPLATES.get(
            main_ohaeng,
            FORTUNE_TEMPLATES["목"]
        )
        
        # 해석 생성
        interpretation = {
            "전체운": self._generate_overall_fortune(
                main_ohaeng,
                ohaeng,
                name
            ),
            "성격": self._generate_personality(
                prediction,
                main_template,
                confidence
            ),
            "조언": self._generate_advice(
                main_ohaeng,
                weak_ohaeng,
                ohaeng
            )
        }
        
        return interpretation
    
    def _generate_overall_fortune(self, main_ohaeng, ohaeng, name):
        """전체 운세 생성"""
        name_str = f"{name}님의 " if name else ""
        
        balance = np.std(list(ohaeng.values()))
        
        if balance < 1.0:
            balance_msg = "오행이 매우 조화롭습니다."
        elif balance < 1.5:
            balance_msg = "오행이 비교적 균형적입니다."
        else:
            balance_msg = "오행의 편차가 있습니다."
        
        text = f"{name_str}사주는 {main_ohaeng} 기운이 강합니다. {balance_msg}"
        
        return text
    
    def _generate_personality(self, prediction, template, confidence):
        """성격 분석 생성"""
        text = f"AI 분석 결과, '{prediction}' 성향을 보입니다. (신뢰도: {confidence*100:.0f}%)\n"
        text += f"특징: {', '.join(template['성격'])}"
        
        return text
    
    def _generate_advice(self, main_ohaeng, weak_ohaeng, ohaeng):
        """조언 생성"""
        advices = []
        
        # 강한 오행 관련 조언
        if main_ohaeng in FORTUNE_TEMPLATES:
            advice = random.choice(FORTUNE_TEMPLATES[main_ohaeng]["운세"])
            advices.append(f"• {main_ohaeng} 기운: {advice}")
        
        # 약한 오행 보완 조언
        weak_advice_map = {
            "목": "창의성과 성장을 위한 활동을 추천합니다.",
            "화": "열정과 사교활동이 도움이 됩니다.",
            "토": "안정적인 계획이 필요합니다.",
            "금": "결단력을 키우는 것이 좋습니다.",
            "수": "유연한 사고와 학습이 중요합니다."
        }
        
        if ohaeng[weak_ohaeng] < 2:
            advices.append(f"• {weak_ohaeng} 보완: {weak_advice_map[weak_ohaeng]}")
        
        return "\n".join(advices)


# 사용 예시
if __name__ == "__main__":
    from model_trainer import ModelTrainer
    from data_preprocessor import DataPreprocessor
    
    # 모델 및 전처리기 로드
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    # 저장된 모델 로드 (또는 새로 학습)
    model_path = "./models/saju_model_랜덤포레스트.pkl"
    model = trainer.load_model(model_path)
    
    if model:
        # API 클라이언트 생성
        api_client = SajuAPIClient()
        
        # 예측기 생성
        predictor = SajuPredictor(model, preprocessor, api_client)
        
        # 사주 예측
        result = predictor.predict_saju(
            birth_year=1990,
            birth_month=5,
            birth_day=15,
            birth_hour=14,
            name="홍길동"
        )
        
        # 결과 출력
        print("\n=== 사주 풀이 결과 ===")
        print(f"\n입력: {result['입력정보']}")
        print(f"\n사주팔자: {result['사주팔자']}")
        print(f"\n오행분석: {result['오행분석']}")
        print(f"\nAI 예측: {result['AI예측']}")
        print(f"\n해석:")
        print(f"  전체운: {result['해석']['전체운']}")
        print(f"  성격: {result['해석']['성격']}")
        print(f"  조언:\n{result['해석']['조언']}")