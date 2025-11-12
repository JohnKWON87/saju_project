"""
2024 하드코딩 → 현재 연도 자동 감지로 수정
"""

import os
import re

def fix_file(filepath, old_pattern, new_code):
    """파일 내용 수정"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 수정 필요 여부 확인
        if old_pattern in content:
            # datetime import 추가 (없으면)
            if 'from datetime import datetime' not in content:
                # import 섹션 찾기
                import_section = content.split('\n\n')[0]
                content = content.replace(
                    import_section,
                    import_section + '\nfrom datetime import datetime'
                )
            
            # 하드코딩된 2024를 현재 연도로 변경
            content = re.sub(
                r"(current_year\s*=\s*)2024",
                r"\1datetime.now().year",
                content
            )
            content = re.sub(
                r"(\(\()2024(\s*-\s*birth_year\))",
                r"\1datetime.now().year\2",
                content
            )
            content = re.sub(
                r"(\(\()2024(\s*-\s*df\['년'\]\))",
                r"\1datetime.now().year\2",
                content
            )
            
            # 파일 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ {filepath} 수정 완료")
            return True
        else:
            print(f"⏭️  {filepath} - 이미 최신 버전")
            return False
            
    except FileNotFoundError:
        print(f"❌ {filepath} 파일 없음")
        return False
    except Exception as e:
        print(f"❌ {filepath} 오류: {e}")
        return False


def main():
    """메인 실행"""
    print("=" * 60)
    print("🔧 2024 하드코딩 수정 스크립트")
    print("=" * 60)
    
    files_to_fix = [
        ('predictor.py', '2024'),
        ('simple_predictor.py', '2024'),
        ('data_preprocessor.py', '2024'),
        ('train_no_scaling.py', '2024')
    ]
    
    fixed_count = 0
    
    for filepath, pattern in files_to_fix:
        if os.path.exists(filepath):
            if fix_file(filepath, pattern, 'datetime.now().year'):
                fixed_count += 1
        else:
            print(f"⚠️  {filepath} 파일이 없습니다")
    
    print("\n" + "=" * 60)
    if fixed_count > 0:
        print(f"✅ {fixed_count}개 파일 수정 완료!")
        print("\n다음 단계:")
        print("1. 모델 재학습: python train_no_scaling.py")
        print("2. 웹앱 실행: streamlit run app_fixed.py")
    else:
        print("✅ 모든 파일이 이미 최신 버전입니다")
    print("=" * 60)


if __name__ == "__main__":
    main()