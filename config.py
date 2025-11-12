"""
설정 파일
API 키 및 프로젝트 설정 관리
"""
from datetime import datetime
SYSTEM_CONFIG = {
    "CURRENT_YEAR": datetime.now().year,  # 자동 갱신
    "BASE_YEAR": 1900,  # 최소 출생 연도
    "MAX_YEAR": datetime.now().year  # 최대 출생 연도
}
# 한국천문연구원 음양력 API 설정 (사용 안 함)
API_CONFIG = {
    "SERVICE_KEY": "여기에_한국천문연구원_API_키를_입력하세요",
    "BASE_URL": "http://apis.data.go.kr/B090041/openapi/service/LrsrCldInfoService",
    "ENDPOINTS": {
        "양력정보": "/getSolCalInfo",
        "음력정보": "/getLunCalInfo",
    }
}

# ✨ Gemini API 설정 (필수)
GEMINI_CONFIG = {
    "API_KEY": "AIzaSyBE6RXk7zh9BzRfDuLy5wt2dlG4z6vW734",
    
    # ✅ 2024년 11월 기준 사용 가능한 모델
    "MODEL": "gemini-2.0-flash",  # 빠르고 무료!
    
    # 대안 모델:
    # "MODEL": "gemini-2.0-flash-lite",  # 더 빠름
    # "MODEL": "gemini-flash-latest",    # 최신 안정 버전
    # "MODEL": "gemini-2.5-flash",       # 더 강력함 (할당량 주의)
    
    "TEMPERATURE": 0.7,
    "MAX_OUTPUT_TOKENS": 1000,
    "TOP_P": 0.95,
    "TOP_K": 40
}

# 데이터 저장 경로
DATA_PATH = {
    "RAW": "./data/raw/",
    "PROCESSED": "./data/processed/",
    "MODEL": "./models/",
    "OUTPUT": "./output/"
}

# 사주 해석 템플릿 (초급자용 간단한 버전)
FORTUNE_TEMPLATES = {
    "목": {
        "성격": ["창의적", "성장지향적", "유연한"],
        "운세": ["새로운 시작에 좋음", "발전의 기회", "인내가 필요"]
    },
    "화": {
        "성격": ["열정적", "활동적", "사교적"],
        "운세": ["활발한 활동 기회", "인기 상승", "감정 조절 필요"]
    },
    "토": {
        "성격": ["안정적", "신뢰있는", "포용력있는"],
        "운세": ["안정된 생활", "신용 상승", "조화로운 관계"]
    },
    "금": {
        "성격": ["결단력있는", "정직한", "원칙적인"],
        "운세": ["결단이 필요한 시기", "변화의 기회", "실행력 발휘"]
    },
    "수": {
        "성격": ["지혜로운", "유연한", "소통하는"],
        "운세": ["지혜로운 선택", "흐름을 따르기", "학습 기회"]
    }
}

# 모델 설정
MODEL_CONFIG = {
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "CV_FOLDS": 5
}

# API 호출 제한
API_LIMITS = {
    "REQUESTS_PER_SECOND": 1,
    "MAX_RETRIES": 3,
    "TIMEOUT": 10
}

def validate_gemini_api_key():
    """Gemini API 키 검증"""
    api_key = GEMINI_CONFIG["API_KEY"]
    
    if not api_key or api_key == "여기에_발급받은_Gemini_API_키를_입력하세요":
        return False, "Gemini API 키가 설정되지 않았습니다."
    
    if len(api_key) < 20:
        return False, "API 키 형식이 올바르지 않습니다."
    
    return True, "API 키가 설정되었습니다."


if __name__ == "__main__":
    print("=" * 50)
    print("설정 파일 검증")
    print("=" * 50)
    
    # Gemini API 키 검증
    is_valid, message = validate_gemini_api_key()
    
    if is_valid:
        print(f"✅ {message}")
        print(f"✅ 모델: {GEMINI_CONFIG['MODEL']}")
    else:
        print(f"❌ {message}")
        print("\n📌 다음 단계를 따라주세요:")
        print("1. https://aistudio.google.com/app/apikey 접속")
        print("2. Google 계정으로 로그인")
        print("3. 'Create API key' 클릭")
        print("4. 생성된 키를 config.py의 GEMINI_CONFIG['API_KEY']에 입력")
        print("\n💡 무료 할당량: 60 requests/minute")
    
    print("\n" + "=" * 50)