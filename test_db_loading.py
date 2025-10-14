# test_db_loading.py

import os
print("=============================================")
print("Vector DB 로컬 로딩 테스트를 시작합니다...")
print(f"현재 작업 경로: {os.getcwd()}")
print("=============================================")

try:
    # search_engine.py 파일이 로딩되면서
    # 전역 범위에 있는 DB 로딩 코드가 자동으로 실행됩니다.
    # 이 import 문이 성공하면 로딩이 성공한 것입니다.
    from app import search_engine

    print("\n[테스트 결과]")
    print("✅ 테스트 스크립트가 성공적으로 완료되었습니다.")
    print("   'search_engine.py' 파일에서 출력된 메시지를 확인하세요.")
    print("   '✅ ... 로딩 성공' 메시지가 모두 보이면 정상입니다.")

except FileNotFoundError as e:
    print("\n[오류 발생!]")
    print(f"❌ DB 파일을 찾을 수 없습니다: {e}")
    print("   'vector_store' 폴더의 경로가 올바른지 확인해주세요.")

except Exception as e:
    print("\n[오류 발생!]")
    print(f"❌ DB 로딩 중 예외가 발생했습니다: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n=============================================")
    print("테스트를 종료합니다.")
    print("=============================================")