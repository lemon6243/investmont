# check_setup.py
# 목적: 라이브러리 설치 검증 + 웹캠 작동 확인 (모듈 0 완료 판정용)

import cv2
import mediapipe as mp
import numpy as np

# 1) 라이브러리 버전 출력 (설치 검증)
print("=" * 40)
print("[설치 검증]")
print("OpenCV 버전:", cv2.__version__)
print("MediaPipe 버전:", mp.__version__)
print("NumPy 버전:", np.__version__)
print("=" * 40)

# 2) 웹캠 열기 시도
print("[웹캠 테스트] 웹캠을 여는 중... 창이 뜨면 성공입니다.")
print("종료하려면 웹캠 창을 클릭한 뒤 키보드에서 q 를 누르세요.")

cap = cv2.VideoCapture(0)  # 0번 = 기본(노트북 내장) 웹캠

if not cap.isOpened():
    print("[실패] 웹캠을 열 수 없습니다. 다른 프로그램(줌, 카메라 앱 등)이")
    print("        웹캠을 쓰고 있지 않은지 확인하세요.")
else:
    while True:
        ret, frame = cap.read()  # 한 프레임 읽기
        if not ret:
            print("[실패] 웹캠에서 영상을 읽지 못했습니다.")
            break

        # 화면 좌우 반전 (거울처럼 보이게 - 수어 동작 확인에 편함)
        frame = cv2.flip(frame, 1)

        # 안내 문구 표시
        cv2.putText(frame, "Press q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("SignBridge - Webcam Test", frame)

        # q 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[완료] 웹캠 테스트 종료.")
