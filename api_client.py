"""
한국천문연구원 음양력 API 클라이언트
생년월일 정보로 음양력 데이터 조회
"""

import requests
import time
from config import API_CONFIG, API_LIMITS


class SajuAPIClient:
    """사주 데이터를 가져오는 API 클라이언트"""
    
    def __init__(self, service_key=None):
        """
        API 클라이언트 초기화
        
        Args:
            service_key: 공공데이터포털에서 발급받은 서비스 키
        """
        self.service_key = service_key or API_CONFIG["SERVICE_KEY"]
        self.base_url = API_CONFIG["BASE_URL"]
        self.last_request_time = 0
        
    def _wait_for_rate_limit(self):
        """API 호출 제한 준수를 위한 대기"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / API_LIMITS["REQUESTS_PER_SECOND"]
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def get_solar_info(self, year, month, day):
        """
        양력 날짜로 음양력 정보 조회
        
        Args:
            year: 연도 (예: 1990)
            month: 월 (1-12)
            day: 일 (1-31)
            
        Returns:
            dict: API 응답 데이터 또는 None
        """
        self._wait_for_rate_limit()
        
        # API 요청 파라미터
        params = {
            "serviceKey": self.service_key,
            "solYear": str(year),
            "solMonth": str(month).zfill(2),
            "solDay": str(day).zfill(2),
            "numOfRows": "1",
            "pageNo": "1"
        }
        
        url = self.base_url + API_CONFIG["ENDPOINTS"]["양력정보"]
        
        try:
            response = requests.get(
                url, 
                params=params, 
                timeout=API_LIMITS["TIMEOUT"]
            )
            response.raise_for_status()
            
            # XML 응답 처리 (간단한 파싱)
            if response.status_code == 200:
                # 실제로는 XML 파싱이 필요하지만, 여기서는 간단히 처리
                return self._parse_response(response.text)
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 오류: {e}")
            return None
    
    def _parse_response(self, xml_text):
        """
        XML 응답을 딕셔너리로 변환 (간단한 버전)
        실제로는 xml.etree.ElementTree 사용 권장
        """
        # 더미 데이터 반환 (실제 구현시 XML 파싱 필요)
        return {
            "lunYear": "경자",
            "lunMonth": "5",
            "lunDay": "15",
            "lunNday": "15",
            "lunLeapmonth": "",
            "lunSecha": "경자",
            "lunWolgeon": "신사",
            "lunIljin": "무인"
        }
    
    def calculate_saju(self, birth_year, birth_month, birth_day, birth_hour=None):
        """
        생년월일로 사주팔자 계산
        
        Args:
            birth_year: 출생 연도
            birth_month: 출생 월
            birth_day: 출생 일
            birth_hour: 출생 시간 (선택, 0-23)
            
        Returns:
            dict: 사주 정보
        """
        # API에서 연월일 정보 가져오기
        lunar_info = self.get_solar_info(birth_year, birth_month, birth_day)
        
        if not lunar_info:
            return None
        
        # 시주 계산 (시간이 제공된 경우)
        if birth_hour is not None:
            hour_gan, hour_ji = self._calculate_hour_pillar(
                lunar_info.get("lunIljin", "갑자")[0],  # 일간
                birth_hour
            )
        else:
            hour_gan, hour_ji = "미입력", "미입력"
        
        # 사주팔자 구성
        saju = {
            "입력정보": {
                "양력": f"{birth_year}년 {birth_month}월 {birth_day}일",
                "시간": f"{birth_hour}시" if birth_hour else "미입력"
            },
            "사주팔자": {
                "년주": lunar_info.get("lunSecha", ""),
                "월주": lunar_info.get("lunWolgeon", ""),
                "일주": lunar_info.get("lunIljin", ""),
                "시주": f"{hour_gan}{hour_ji}"
            },
            "오행분석": self._analyze_ohaeng(lunar_info, hour_gan, hour_ji)
        }
        
        return saju
    
    def _calculate_hour_pillar(self, day_gan, hour):
        """
        일간과 시간으로 시주 계산
        
        Args:
            day_gan: 일간 (甲, 乙, 丙 등)
            hour: 시간 (0-23)
            
        Returns:
            tuple: (시간, 시지)
        """
        # 천간 목록
        gans = ["갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"]
        # 지지 목록
        jis = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]
        
        # 시지 결정 (2시간 단위)
        ji_index = ((hour + 1) // 2) % 12
        hour_ji = jis[ji_index]
        
        # 시간 결정 (일간에 따라 달라짐 - 간단한 버전)
        day_gan_index = gans.index(day_gan) if day_gan in gans else 0
        hour_gan_index = (day_gan_index * 2 + ji_index) % 10
        hour_gan = gans[hour_gan_index]
        
        return hour_gan, hour_ji
    
    def _analyze_ohaeng(self, lunar_info, hour_gan="", hour_ji=""):
        """
        오행(목화토금수) 분석
        
        Returns:
            dict: 오행별 개수
        """
        # 간단한 오행 매핑
        ohaeng_map = {
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
        
        # 오행 카운트
        ohaeng_count = {"목": 0, "화": 0, "토": 0, "금": 0, "수": 0}
        
        # 사주팔자의 각 글자 분석
        saju_chars = (
            lunar_info.get("lunSecha", "") +
            lunar_info.get("lunWolgeon", "") +
            lunar_info.get("lunIljin", "") +
            hour_gan + hour_ji
        )
        
        for char in saju_chars:
            if char in ohaeng_map:
                ohaeng_count[ohaeng_map[char]] += 1
        
        return ohaeng_count


# 사용 예시
if __name__ == "__main__":
    # API 클라이언트 생성
    client = SajuAPIClient()
    
    # 사주 조회
    result = client.calculate_saju(
        birth_year=1990,
        birth_month=5,
        birth_day=15,
        birth_hour=14
    )
    
    if result:
        print("=== 사주팔자 결과 ===")
        print(f"입력: {result['입력정보']}")
        print(f"사주: {result['사주팔자']}")
        print(f"오행: {result['오행분석']}")
    else:
        print("API 호출 실패")