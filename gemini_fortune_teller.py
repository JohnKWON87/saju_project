"""
Gemini API를 활용한 사주 해석
자연스러운 AI 사주풀이
"""

import google.generativeai as genai
from saju_calculator import SajuCalculator
from config import GEMINI_CONFIG
import time
from datetime import datetime


class GeminiSajuPredictor:
    """Gemini API 기반 사주 예측기"""
    
    def __init__(self, api_key=None):
        """
        Gemini 예측기 초기화
        
        Args:
            api_key: Gemini API 키 (None이면 config에서 가져옴)
        """
        self.api_key = api_key or GEMINI_CONFIG["API_KEY"]
        
        # API 키 검증
        if not self.api_key or len(self.api_key) < 20:
            raise ValueError(
                "Gemini API 키가 설정되지 않았습니다.\n"
                "config.py에서 GEMINI_CONFIG['API_KEY']를 설정하세요.\n"
                "API 키 발급: https://aistudio.google.com/app/apikey"
            )
        
        # Gemini 설정
        genai.configure(api_key=self.api_key)
        
        # ✅ 수정: 사용 가능한 모델 목록 확인
        available_models = [
            "gemini-pro",           # 안정적인 기본 모델
            "gemini-1.5-pro",       # 최신 모델 (사용 가능할 경우)
            "gemini-1.5-flash"      # 빠른 모델 (사용 가능할 경우)
        ]
        
        # 설정된 모델 또는 기본 모델 사용
        model_name = GEMINI_CONFIG.get("MODEL", "gemini-pro")
        
        # gemini-1.5-flash가 설정되어 있으면 gemini-pro로 변경
        if "1.5" in model_name:
            print(f"⚠️  {model_name}는 현재 지원되지 않을 수 있습니다. gemini-pro를 사용합니다.")
            model_name = "gemini-pro"
        
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": GEMINI_CONFIG.get("TEMPERATURE", 0.7),
                    "top_p": GEMINI_CONFIG.get("TOP_P", 0.95),
                    "top_k": GEMINI_CONFIG.get("TOP_K", 40),
                    "max_output_tokens": GEMINI_CONFIG.get("MAX_OUTPUT_TOKENS", 1000),
                }
            )
            print(f"✅ Gemini API 초기화 완료 (모델: {model_name})")
            
        except Exception as e:
            print(f"❌ 모델 초기화 오류: {e}")
            print("기본 gemini-pro 모델을 사용합니다.")
            self.model = genai.GenerativeModel("gemini-pro")
        
        self.calculator = SajuCalculator()
    
    def predict_saju(self, birth_year, birth_month, birth_day, birth_hour=None, name=None):
        """
        생년월일로 사주 예측 (Gemini AI 사용)
        
        Args:
            birth_year: 출생 연도
            birth_month: 출생 월
            birth_day: 출생 일
            birth_hour: 출생 시간 (선택)
            name: 이름 (선택)
            
        Returns:
            dict: 사주 예측 결과
        """
        print(f"\n=== Gemini AI 사주 풀이 시작 ===")
        print(f"입력: {birth_year}년 {birth_month}월 {birth_day}일", end="")
        if birth_hour is not None:
            print(f" {birth_hour}시")
        else:
            print(" (시간 미입력)")
        
        # 1. 사주 계산
        saju_data = self.calculator.calculate_saju(
            birth_year, birth_month, birth_day, birth_hour
        )
        
        # 2. Gemini 프롬프트 생성
        prompt = self._create_prompt(saju_data, name)
        
        # 3. Gemini API 호출
        try:
            print("🤖 Gemini AI 분석 중...")
            response = self.model.generate_content(prompt)
            interpretation = response.text
            print("✅ 분석 완료!")
            
        except Exception as e:
            print(f"❌ Gemini API 오류: {e}")
            interpretation = self._fallback_interpretation(saju_data)
        
        # 4. 결과 구성
        result = {
            "입력정보": saju_data['입력정보'],
            "사주팔자": saju_data['사주팔자'],
            "오행분석": saju_data['오행분석'],
            "AI예측": {
                "모델": "Gemini Pro",
                "해석": interpretation
            }
        }
        
        return result
    
    def _create_prompt(self, saju_data, name):
        """Gemini용 프롬프트 생성"""
        
        name_str = f"{name}님의 " if name else ""
        current_year = datetime.now().year
        
        # 주요 오행 분석
        ohaeng = saju_data['오행분석']
        main_ohaeng = max(ohaeng, key=ohaeng.get)
        weak_ohaeng = min(ohaeng, key=ohaeng.get)
        
        prompt = f"""
당신은 30년 경력의 전문 사주 명리학자입니다. 
다음 사주 정보를 바탕으로 상세하고 따뜻한 어조로 사주 풀이를 해주세요.

【 기본 정보 】
- 생년월일: {saju_data['입력정보']['양력']}
- 시간: {saju_data['입력정보']['시간']}

【 사주팔자 】
- 년주(年柱): {saju_data['사주팔자']['년주']}
- 월주(月柱): {saju_data['사주팔자']['월주']}
- 일주(日柱): {saju_data['사주팔자']['일주']}
- 시주(時柱): {saju_data['사주팔자']['시주']}

【 오행 분석 】
- 목(木): {ohaeng['목']}개
- 화(火): {ohaeng['화']}개
- 토(土): {ohaeng['토']}개
- 금(金): {ohaeng['금']}개
- 수(水): {ohaeng['수']}개

→ 주요 오행: {main_ohaeng}
→ 부족 오행: {weak_ohaeng}

【 요청사항 】
다음 4가지 항목으로 구조화된 풀이를 작성해주세요:

1. **전체운** (150-200자)
   - {name_str}사주의 전반적인 기운과 흐름
   - 오행의 균형 상태 평가
   
2. **성격 분석** (200-300자)
   - 타고난 성향과 장단점
   - 인간관계에서의 특징
   - 강점 3가지
   
3. **{current_year}년 운세** (150-200자)
   - 올해 주의해야 할 점
   - 좋은 기회가 올 시기
   
4. **인생 조언** (100-150자)
   - 부족한 오행({weak_ohaeng})을 보완하는 방법
   - 강한 오행({main_ohaeng})을 활용하는 법
   - 실천 가능한 구체적 조언

【 작성 원칙 】
- 따뜻하고 희망적인 어조 사용
- 부정적 표현보다 긍정적 조언 중심
- 전문 용어는 쉽게 풀어서 설명
- 각 항목은 제목을 포함해서 작성
- 이모지 사용 금지
"""
        
        return prompt
    
    def _fallback_interpretation(self, saju_data):
        """API 실패시 기본 해석"""
        ohaeng = saju_data['오행분석']
        main_ohaeng = max(ohaeng, key=ohaeng.get)
        current_year = datetime.now().year
        
        return f"""
**전체운**
사주에 {main_ohaeng} 기운이 강하게 나타납니다.

**성격 분석**
Gemini API 연결 오류로 상세 분석을 제공할 수 없습니다.
API 키를 확인해주세요.

**{current_year}년 운세**
API 오류

**인생 조언**
config.py에서 Gemini API 키를 확인하세요.
발급 링크: https://aistudio.google.com/app/apikey
"""


# 테스트 코드
if __name__ == "__main__":
    try:
        predictor = GeminiSajuPredictor()
        
        result = predictor.predict_saju(
            birth_year=1990,
            birth_month=5,
            birth_day=15,
            birth_hour=14,
            name="홍길동"
        )
        
        print("\n=== 사주 풀이 결과 ===")
        print(f"\n입력: {result['입력정보']}")
        print(f"\n사주팔자: {result['사주팔자']}")
        print(f"\n오행분석: {result['오행분석']}")
        print(f"\nAI 해석:\n{result['AI예측']['해석']}")
        
    except ValueError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()