# check_lr.py
# 목적: MediaPipe -> 공통포맷 변환에서 좌/우(swap_lr)와 flip 설정이 맞는지 눈으로 검증
# 사용법: 실행 후 "오른손만" 들어보세요. 화면 안내대로 어느 손에 좌표가 잡히는지 확인.

import cv2
import mediapipe as mp
import numpy as np
from keypoint_schema import mediapipe_to_common, N_POSE

# ------------------------------------------------------------
# 여기 두 값을 바꿔가며 테스트합니다.
#   USE_FLIP   : 웹캠 좌우반전 켤지 (거울모드)
#   SWAP_LR    : mediapipe_to_common의 좌/우 교체 여부
# ------------------------------------------------------------
USE_FLIP = False    # 먼저 flip을 끄고 테스트 (권장)
SWAP_LR  = False

mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[실패] 웹캠을 열 수 없습니다.")
else:
    print("=" * 55)
    print("[좌/우 검증] 오른손만 들어보세요.")
    print(f"  현재 설정: USE_FLIP={USE_FLIP}, SWAP_LR={SWAP_LR}")
    print("  화면 상단 텍스트에서 R-hand 값이 켜지면 정답입니다.")
    print("  종료: q")
    print("=" * 55)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if USE_FLIP:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # 공통포맷으로 변환 (50,2)
            common = mediapipe_to_common(results, w, h, swap_lr=SWAP_LR)

            # 공통포맷 구조: [0:8]=상체, [8:29]=왼손, [29:50]=오른손
            lhand = common[N_POSE : N_POSE + 21]        # 공통포맷 '왼손'
            rhand = common[N_POSE + 21 : N_POSE + 42]   # 공통포맷 '오른손'

            # 좌표가 0이 아니면 그 손이 "잡혔다"고 판단
            l_on = np.any(lhand != 0.0)
            r_on = np.any(rhand != 0.0)

            txt = f"[common] L-hand:{'ON ' if l_on else 'off'}  R-hand:{'ON ' if r_on else 'off'}"
            color = (0, 255, 0) if r_on and not l_on else (0, 255, 255)
            cv2.putText(image, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(image, f"FLIP={USE_FLIP} SWAP_LR={SWAP_LR}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.putText(image, "Raise your RIGHT hand only. q=quit", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("SignBridge - L/R Check", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
