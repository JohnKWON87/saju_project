"""
Gemini API 키 진단 스크립트
"""

import google.generativeai as genai
from config import GEMINI_CONFIG

print("=" * 60)
print("Gemini API 키 진단 시작")
print("=" * 60)

# 1. API 키 확인
api_key = GEMINI_CONFIG["API_KEY"]
print(f"\n1. API 키 길이: {len(api_key)}")
print(f"   API 키 시작: {api_key[:10]}...")
print(f"   API 키 끝: ...{api_key[-10:]}")

# 2. API 키 형식 확인
if api_key.startswith("AIza"):
    print("   ✅ API 키 형식이 올바릅니다.")
else:
    print("   ❌ API 키 형식이 이상합니다. 'AIza'로 시작해야 합니다.")

# 3. API 연결 테스트
print("\n2. API 연결 테스트 중...")
try:
    genai.configure(api_key=api_key)
    print("   ✅ API 키 설정 완료")
    
    # 4. 사용 가능한 모델 목록 확인
    print("\n3. 사용 가능한 모델 확인 중...")
    models = genai.list_models()
    
    print("   사용 가능한 모델:")
    gemini_models = []
    for model in models:
        if 'gemini' in model.name.lower():
            gemini_models.append(model.name)
            print(f"   - {model.name}")
    
    if not gemini_models:
        print("   ❌ Gemini 모델을 찾을 수 없습니다!")
    
    # 5. 실제 API 호출 테스트
    print("\n4. 실제 API 호출 테스트 중...")
    
    # gemini-pro 테스트
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("안녕하세요")
        print("   ✅ gemini-pro 작동 확인!")
        print(f"   응답: {response.text[:50]}...")
        
    except Exception as e:
        print(f"   ❌ gemini-pro 오류: {e}")
        
        # 다른 모델 시도
        if gemini_models:
            print(f"\n   대체 모델 시도: {gemini_models[0]}")
            try:
                # models/gemini-pro -> gemini-pro 형식으로 변환
                model_name = gemini_models[0].split('/')[-1]
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("안녕하세요")
                print(f"   ✅ {model_name} 작동 확인!")
                print(f"   응답: {response.text[:50]}...")
                print(f"\n   💡 config.py의 MODEL을 '{model_name}'으로 변경하세요!")
                
            except Exception as e2:
                print(f"   ❌ {model_name} 오류: {e2}")
    
    print("\n" + "=" * 60)
    print("진단 완료!")
    print("=" * 60)
    
except Exception as e:
    print(f"   ❌ API 연결 실패: {e}")
    print("\n💡 해결 방법:")
    print("1. API 키를 다시 발급받으세요: https://aistudio.google.com/app/apikey")
    print("2. 발급 시 'Generative Language API' 권한이 있는지 확인")
    print("3. 무료 할당량이 남아있는지 확인")