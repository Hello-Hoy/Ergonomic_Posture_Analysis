import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중간 점 (꼭짓점)
    c = np.array(c)  # 세 번째 점

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def analyze_head_posture(landmarks):
    # MediaPipe 랜드마크 인덱스
    RIGHT_SHOULDER = 12
    LEFT_SHOULDER = 11
    RIGHT_EAR = 8
    LEFT_EAR = 7

    # 올바른 랜드마크 인덱스로 접근
    shoulder_r = landmarks[RIGHT_SHOULDER]  # [x, y, z] 좌표
    shoulder_l = landmarks[LEFT_SHOULDER]   # [x, y, z] 좌표
    ear_r = landmarks[RIGHT_EAR]           # [x, y, z] 좌표

    # 어깨 중심점 계산 (x, y 좌표만 사용)
    shoulder_center = [
        (shoulder_l[0] + shoulder_r[0]) / 2,
        (shoulder_l[1] + shoulder_r[1]) / 2
    ]

    # CVA 계산을 위한 수평선상의 점 생성
    horizontal_point = [shoulder_center[0] - 1, shoulder_center[1]]  # 어깨 중심에서 왼쪽으로 수평

    # 2D 평면에서 각도 계산 (x, y 좌표만 사용)
    cva = calculate_angle(horizontal_point, shoulder_center, ear_r[:2])

    if cva < 50:
        return 'FORWARD_HEAD', f"거북목 주의: 현재 각도 {int(cva)}도. 귀를 어깨선과 맞추세요."
    return 'GOOD', f"좋은 자세입니다. 현재 각도: {int(cva)}도"


def analyze_shoulder_posture(landmarks):
    # MediaPipe 랜드마크 인덱스
    RIGHT_SHOULDER = 12
    LEFT_SHOULDER = 11
    RIGHT_HIP = 24
    LEFT_HIP = 23

    # 올바른 랜드마크 인덱스로 접근
    shoulder_r = landmarks[RIGHT_SHOULDER]  # [x, y, z] 좌표
    shoulder_l = landmarks[LEFT_SHOULDER]   # [x, y, z] 좌표
    hip_r = landmarks[RIGHT_HIP]           # [x, y, z] 좌표
    hip_l = landmarks[LEFT_HIP]            # [x, y, z] 좌표

    # 어깨와 엉덩이의 z-좌표(깊이) 평균 계산
    shoulder_z_avg = (shoulder_l[2] + shoulder_r[2]) / 2
    hip_z_avg = (hip_l[2] + hip_r[2]) / 2

    # 임계값 설정 (실험적으로 조정 필요)
    slump_threshold = -0.1  # 어깨가 엉덩이보다 앞으로 나온 정도

    if shoulder_z_avg < hip_z_avg + slump_threshold:
        return 'SLUMPED', "어깨가 구부정합니다. 가슴을 펴고 어깨를 뒤로 젖히세요."

    return 'GOOD', "바른 어깨 자세입니다."