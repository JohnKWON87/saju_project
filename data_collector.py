"""
API 없이 사주 데이터 수집
자체 계산 라이브러리 사용
"""

import pandas as pd
import random
from saju_calculator import SajuCalculator
from config import FORTUNE_TEMPLATES
import os


class DataCollectorV2:
    """API 없이 사주 데이터 수집"""
    
    def __init__(self):
        """데이터 수집기 초기화"""
        self.calculator = SajuCalculator()
        self.collected_data = []
    
    def generate_random_birthdate(self, start_year=1950, end_year=2023):
        """랜덤 생년월일 생성"""
        year = random.randint(start_year, end_year)
        month = random.randint(1, 12)
        
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        if month == 2 and year % 4 == 0:
            if year % 100 != 0 or year % 400 == 0:
                days_in_month[1] = 29
        
        day = random.randint(1, days_in_month[month - 1])
        hour = random.randint(0, 23) if random.random() > 0.3 else None
        
        return year, month, day, hour
    
    def collect_sample_data(self, num_samples=100):
        """
        샘플 데이터 수집 (API 불필요)
        
        Args:
            num_samples: 수집할 샘플 수
            
        Returns:
            list: 수집된 데이터
        """
        print(f"=== 데이터 수집 시작 ({num_samples}개) ===")
        print("✓ API 불필요 - 자체 계산 사용")
        
        for i in range(num_samples):
            try:
                # 랜덤 생년월일 생성
                year, month, day, hour = self.generate_random_birthdate()
                
                # 자체 계산으로 사주 데이터 생성
                saju_data = self.calculator.calculate_saju(year, month, day, hour)
                
                # 운세 레이블 생성
                fortune_label = self._generate_fortune_label(
                    saju_data['오행분석']
                )
                
                # 데이터 저장
                record = {
                    "ID": i + 1,
                    "년": year,
                    "월": month,
                    "일": day,
                    "시": hour if hour is not None else -1,
                    "년주": saju_data['사주팔자']['년주'],
                    "월주": saju_data['사주팔자']['월주'],
                    "일주": saju_data['사주팔자']['일주'],
                    "시주": saju_data['사주팔자']['시주'],
                    "목": saju_data['오행분석']['목'],
                    "화": saju_data['오행분석']['화'],
                    "토": saju_data['오행분석']['토'],
                    "금": saju_data['오행분석']['금'],
                    "수": saju_data['오행분석']['수'],
                    "주요오행": fortune_label['주요오행'],
                    "성격유형": fortune_label['성격'],
                    "운세유형": fortune_label['운세']
                }
                
                self.collected_data.append(record)
                
                if (i + 1) % 20 == 0:
                    print(f"진행: {i + 1}/{num_samples} ({(i+1)/num_samples*100:.1f}%)")
                
            except Exception as e:
                print(f"샘플 {i+1} 수집 오류: {e}")
                continue
        
        print(f"✓ 수집 완료! 총 {len(self.collected_data)}개")
        return self.collected_data
    
    def _generate_fortune_label(self, ohaeng_analysis):
        """오행 기반 운세 레이블 생성"""
        main_ohaeng = max(ohaeng_analysis, key=ohaeng_analysis.get)
        templates = FORTUNE_TEMPLATES.get(main_ohaeng, FORTUNE_TEMPLATES["목"])
        
        return {
            "주요오행": main_ohaeng,
            "성격": random.choice(templates["성격"]),
            "운세": random.choice(templates["운세"])
        }
    
    def save_to_csv(self, filename="saju_dataset.csv"):
        """CSV로 저장"""
        if not self.collected_data:
            print("저장할 데이터가 없습니다.")
            return
        
        df = pd.DataFrame(self.collected_data)
        os.makedirs("./data", exist_ok=True)
        filepath = f"./data/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"\n=== 데이터 저장 완료 ===")
        print(f"파일: {filepath}")
        print(f"레코드 수: {len(df)}")
        print(f"\n데이터 미리보기:")
        print(df.head())
        
        return df
    
    def load_from_csv(self, filename="saju_dataset.csv"):
        """CSV 불러오기"""
        try:
            filepath = f"./data/{filename}"
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            print(f"✓ {filename} 불러오기 성공 ({len(df)}개)")
            return df
        except FileNotFoundError:
            print(f"✗ {filename} 파일 없음")
            return None


# 사용 예시
if __name__ == "__main__":
    # 데이터 수집기 생성 (API 불필요!)
    collector = DataCollectorV2()
    
    # 데이터 수집
    collector.collect_sample_data(num_samples=100)
    
    # CSV 저장
    df = collector.save_to_csv("saju_dataset.csv")
    
    print("\n✓ API 없이 데이터 수집 완료!")