# extract_keypoints.py
# 모듈 1: 웹캠 영상에서 손 + 자세 + 얼굴 키포인트를 실시간으로 추출하고 화면에 그리기
# 목적: 수어 인식 엔진의 "눈" - 관절 좌표(키포인트)를 뽑아내는 단계
# 방식: MediaPipe Holistic (손 21x2 + 자세 33 + 얼굴 랜드마크를 한 번에 처리)

import cv2
import mediapipe as mp
import numpy as np

# ------------------------------------------------------------
# 1) MediaPipe 준비
# ------------------------------------------------------------
mp_holistic = mp.solutions.holistic      # 손+자세+얼굴 통합 모델
mp_drawing = mp.solutions.drawing_utils  # 화면에 점/선을 그려주는 도구
mp_drawing_styles = mp.solutions.drawing_styles  # 기본 그리기 스타일

print("=" * 50)
print("[모듈 1] 키포인트 추출 시작")
print("웹캠 창이 뜨면 손/얼굴/상반신에 점과 선이 그려집니다.")
print("종료: 웹캠 창을 클릭한 뒤 키보드 q")
print("=" * 50)


# ------------------------------------------------------------
# 2) 감지된 키포인트 개수를 세어주는 함수 (검증용)
#    - 손/자세/얼굴이 실제로 잡히는지 숫자로 확인하기 위함
# ------------------------------------------------------------
def count_landmarks(results):
    left_hand = 0
    right_hand = 0
    pose = 0
    face = 0

    if results.left_hand_landmarks:
        left_hand = len(results.left_hand_landmarks.landmark)
    if results.right_hand_landmarks:
        right_hand = len(results.right_hand_landmarks.landmark)
    if results.pose_landmarks:
        pose = len(results.pose_landmarks.landmark)
    if results.face_landmarks:
        face = len(results.face_landmarks.landmark)

    return left_hand, right_hand, pose, face


# ------------------------------------------------------------
# 3) 웹캠 열기
# ------------------------------------------------------------
cap = cv2.VideoCapture(0)  # 0 = 노트북 내장 웹캠

if not cap.isOpened():
    print("[실패] 웹캠을 열 수 없습니다. 다른 앱(줌/카메라)이 웹캠을 쓰고 있지 않은지 확인하세요.")
else:
    # Holistic 모델을 with 블록으로 실행 (자동으로 자원 정리)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,   # 이 값 이상 확신할 때만 감지
        min_tracking_confidence=0.5,    # 추적 유지 최소 확신도
        model_complexity=1              # 0=빠름/낮은정확도, 1=보통, 2=정확/느림
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[실패] 웹캠에서 프레임을 읽지 못했습니다.")
                break

            # (1) 좌우 반전 (거울처럼)
            frame = cv2.flip(frame, 1)

            # (2) MediaPipe는 RGB를 쓰므로 BGR->RGB 변환
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False   # 처리 속도 최적화

            # (3) 키포인트 감지 실행
            results = holistic.process(image_rgb)

            # (4) 다시 그리기 위해 BGR로 복원
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # ----------------------------------------------------
            # (5) 감지 결과를 화면에 그리기
            # ----------------------------------------------------
            # 얼굴 그물망 (비수지신호: 표정/입모양 확인용)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )

            # 상반신 자세
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style()
                )

            # 왼손
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_hand_connections_style()
                )

            # 오른손
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_hand_connections_style()
                )

            # ----------------------------------------------------
            # (6) 감지된 키포인트 개수를 화면에 텍스트로 표시 (검증용)
            # ----------------------------------------------------
            lh, rh, pose, face = count_landmarks(results)
            info = f"L-Hand:{lh}  R-Hand:{rh}  Pose:{pose}  Face:{face}"
            cv2.putText(image, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "Press q to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # (7) 화면 출력
            cv2.imshow("SignBridge - Keypoint Extraction (Module 1)", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[완료] 모듈 1 키포인트 추출 종료.")
