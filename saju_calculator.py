"""
자체 사주 계산 라이브러리
API 없이 생년월일로 사주팔자 계산
"""

from datetime import datetime, timedelta


class SajuCalculator:
    """API 없이 사주팔자 계산"""
    
    # 천간 (10개)
    GAN = ["갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"]
    
    # 지지 (12개)
    JI = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]
    
    # 오행 매핑
    OHAENG_MAP = {
        "갑": "목", "을": "목",
        "병": "화", "정": "화",
        "무": "토", "기": "토",
        "경": "금", "신": "금",
        "임": "수", "계": "수",
        "인": "목", "묘": "목",
        "사": "화", "오": "화",
        "진": "토", "술": "토", "축": "토", "미": "토",
        "신": "금", "유": "금",
        "해": "수", "자": "수"
    }
    
    # 월지 매핑 (양력 기준 근사치)
    MONTH_JI = {
        1: "축", 2: "인", 3: "묘", 4: "진", 5: "사", 6: "오",
        7: "미", 8: "신", 9: "유", 10: "술", 11: "해", 12: "자"
    }
    
    def __init__(self):
        """사주 계산기 초기화"""
        # 기준년도 (1984년 = 갑자년)
        self.base_year = 1984
        self.base_gan_index = 0  # 갑
        self.base_ji_index = 0   # 자
    
    def calculate_saju(self, year, month, day, hour=None):
        """
        생년월일로 사주팔자 계산
        
        Args:
            year: 출생 연도
            month: 출생 월
            day: 출생 일
            hour: 출생 시간 (0-23, 선택)
            
        Returns:
            dict: 사주팔자 정보
        """
        # 연주 계산
        year_gan, year_ji = self._calculate_year_pillar(year)
        
        # 월주 계산
        month_gan, month_ji = self._calculate_month_pillar(year, month, year_gan)
        
        # 일주 계산
        day_gan, day_ji = self._calculate_day_pillar(year, month, day)
        
        # 시주 계산
        if hour is not None:
            hour_gan, hour_ji = self._calculate_hour_pillar(day_gan, hour)
        else:
            hour_gan, hour_ji = "미입력", "미입력"
        
        # 사주팔자
        saju = {
            "년주": f"{year_gan}{year_ji}",
            "월주": f"{month_gan}{month_ji}",
            "일주": f"{day_gan}{day_ji}",
            "시주": f"{hour_gan}{hour_ji}"
        }
        
        # 오행 분석
        ohaeng = self._analyze_ohaeng(year_gan, year_ji, month_gan, month_ji,
                                       day_gan, day_ji, hour_gan, hour_ji)
        
        return {
            "입력정보": {
                "양력": f"{year}년 {month}월 {day}일",
                "시간": f"{hour}시" if hour is not None else "미입력"
            },
            "사주팔자": saju,
            "오행분석": ohaeng
        }
    
    def _calculate_year_pillar(self, year):
        """연주 계산"""
        year_diff = year - self.base_year
        
        gan_index = (self.base_gan_index + year_diff) % 10
        ji_index = (self.base_ji_index + year_diff) % 12
        
        return self.GAN[gan_index], self.JI[ji_index]
    
    def _calculate_month_pillar(self, year, month, year_gan):
        """월주 계산"""
        # 월지
        month_ji = self.MONTH_JI.get(month, "자")
        
        # 월간 (년간에 따라 결정)
        year_gan_index = self.GAN.index(year_gan)
        
        # 간단한 월간 계산 (정확하지 않을 수 있음)
        month_gan_index = (year_gan_index * 2 + month - 1) % 10
        month_gan = self.GAN[month_gan_index]
        
        return month_gan, month_ji
    
    def _calculate_day_pillar(self, year, month, day):
        """일주 계산 (율리우스일 기반)"""
        # 기준일 (1984-01-01 = 갑자일로 가정)
        base_date = datetime(1984, 1, 1)
        target_date = datetime(year, month, day)
        
        # 일수 차이
        day_diff = (target_date - base_date).days
        
        gan_index = (self.base_gan_index + day_diff) % 10
        ji_index = (self.base_ji_index + day_diff) % 12
        
        return self.GAN[gan_index], self.JI[ji_index]
    
    def _calculate_hour_pillar(self, day_gan, hour):
        """시주 계산"""
        # 시지 (2시간 단위)
        ji_index = ((hour + 1) // 2) % 12
        hour_ji = self.JI[ji_index]
        
        # 시간 (일간에 따라 결정)
        day_gan_index = self.GAN.index(day_gan)
        hour_gan_index = (day_gan_index * 2 + ji_index) % 10
        hour_gan = self.GAN[hour_gan_index]
        
        return hour_gan, hour_ji
    
    def _analyze_ohaeng(self, year_gan, year_ji, month_gan, month_ji,
                        day_gan, day_ji, hour_gan, hour_ji):
        """오행 분석"""
        ohaeng_count = {"목": 0, "화": 0, "토": 0, "금": 0, "수": 0}
        
        # 모든 글자 수집
        chars = [year_gan, year_ji, month_gan, month_ji, 
                 day_gan, day_ji]
        
        if hour_gan != "미입력":
            chars.extend([hour_gan, hour_ji])
        
        # 오행 카운트
        for char in chars:
            if char in self.OHAENG_MAP:
                ohaeng = self.OHAENG_MAP[char]
                ohaeng_count[ohaeng] += 1
        
        return ohaeng_count
    
    def get_ohaeng_explanation(self, ohaeng_count):
        """오행 설명"""
        total = sum(ohaeng_count.values())
        
        explanations = []
        for ohaeng, count in ohaeng_count.items():
            ratio = count / total * 100 if total > 0 else 0
            explanations.append(f"{ohaeng}: {count}개 ({ratio:.1f}%)")
        
        return "\n".join(explanations)


# 사용 예시
if __name__ == "__main__":
    calculator = SajuCalculator()
    
    # 사주 계산
    result = calculator.calculate_saju(
        year=1990,
        month=5,
        day=15,
        hour=14
    )
    
    print("=== 사주팔자 계산 결과 ===")
    print(f"\n입력: {result['입력정보']}")
    print(f"\n사주팔자:")
    for key, value in result['사주팔자'].items():
        print(f"  {key}: {value}")
    
    print(f"\n오행분석:")
    print(calculator.get_ohaeng_explanation(result['오행분석']))
    
    print(f"\n오행 데이터: {result['오행분석']}")