import numpy as np

"""
인체공학 임계값(기본값)과 참고 가이드
- Craniovertebral Angle(CVA) 50° 미만: 거북목(FHP) 지표로 자주 사용됨. (예: Ruivo et al., 2014; 물리치료 논문)
- 팔꿈치 각도: 타이핑 시 90°±10° 권장. (Cornell University Ergonomics; OSHA/NIOSH)
- 모니터 눈높이: 눈높이와 거의 같거나 약간 낮게; 고개를 숙인 자세는 피함. (Cornell Ergonomics)
- 라운드 숄더: 어깨가 엉덩이보다 카메라 쪽(z가 더 작다고 가정)으로 과도하게 돌출되면 경고.
주의: 노트북 전면 카메라 환경에서는 무릎/허리 관측이 어려워 '등/허리 각도' 분석(analyze_back_posture)은 비활성화/제거되었습니다.
참고: MediaPipe Pose의 z축은 카메라 방향이 음수로 가까워질 수 있습니다. 카메라, 렌즈, 보정 상태에 따라 반대가 될 수 있어 사용 환경에서 튜닝이 필요합니다.
"""

# 임계값 상수 (필요 시 사용자 환경에 맞게 조정)
CVA_MIN_DEG = 70           # CVA 50° 미만이면 거북목 의심
ELBOW_MIN_DEG = 80         # 팔꿈치 권장 범위 하한
ELBOW_MAX_DEG = 100        # 팔꿈치 권장 범위 상한
EYE_DOWN_NOSE_DIFF = 0.04  # 코와 눈 y차이가 이 값보다 크면 "고개 숙임"으로 판단(정규화 좌표)
SLUMP_Z_THRESHOLD = -0.25  # 라운드 숄더 z차 임계값(환경에 따라 조정)


def calculate_angle(a, b, c):
    """
    세 점 a-b-c로 이루어진 각도(도 단위)를 계산합니다.
    - 입력: a, b, c는 [x, y] 또는 [x, y, z] 형태의 좌표 리스트/배열. b가 꼭짓점입니다.
    - 반환: 0~180 범위의 각도(float).
    """
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중간 점 (꼭짓점)
    c = np.array(c)  # 세 번째 점

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def analyze_head_posture(landmarks):
    """
    머리/목 정렬(CVA) 분석.
    입력: landmarks - MediaPipe Pose 33개 랜드마크의 [x,y,z] 리스트.
    반환: (status, message) 튜플. status는 'GOOD' 또는 'FORWARD_HEAD'.
    """
    # MediaPipe 랜드마크 인덱스
    RIGHT_SHOULDER = 12
    LEFT_SHOULDER = 11
    RIGHT_EAR = 8

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

    if cva < CVA_MIN_DEG:
        return 'FORWARD_HEAD', f"거북목 주의: CVA {int(cva)}° < {CVA_MIN_DEG}°. 귀를 어깨선과 맞추세요."
    return 'GOOD', f"목 자세 양호: CVA {int(cva)}°"


def analyze_shoulder_posture(landmarks):
    """
    어깨 말림(라운드 숄더) 추정.
    - 입력: landmarks - MediaPipe Pose 33개 랜드마크의 [x,y,z] 리스트.
    - 방법: 어깨 평균 z와 엉덩이 평균 z를 비교해 어깨가 카메라 쪽으로 과도하게 돌출되었는지 확인.
    - 반환: (status, message) 튜플. status는 'GOOD' 또는 'SLUMPED'.
    주의: z축 방향/부호는 장비에 따라 달라 환경 보정이 필요할 수 있습니다.
    """
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

    # 임계값 설정 (환경에 따라 조정 필요)
    if shoulder_z_avg < hip_z_avg + SLUMP_Z_THRESHOLD:
        return 'SLUMPED', "어깨가 말려 있습니다. 가슴을 펴고 어깨를 뒤로 젖히세요."

    return 'GOOD', "어깨 자세 양호"




def analyze_elbow_posture(landmarks):
    """
    팔꿈치 각도 분석.
    - 입력: landmarks - MediaPipe Pose 33개 랜드마크 [x,y,z] 리스트.
    - 방법: 어깨-팔꿈치-손목 각도를 좌/우 각각 계산하여 권장 범위 비교.
    - 반환: (status, message). 'GOOD' 또는 'ELBOW_ANGLE_ISSUE'.
    """
    # 팔꿈치 각도: 어깨-팔꿈치-손목
    # 웹캠 좌우 반전으로 인해 RIGHT와 LEFT를 교차하여 사용합니다.
    LEFT_SHOULDER, RIGHT_SHOULDER = 12, 11
    LEFT_ELBOW, RIGHT_ELBOW = 14, 13
    LEFT_WRIST, RIGHT_WRIST = 16, 15

    left_angle = calculate_angle(landmarks[LEFT_SHOULDER][:2], landmarks[LEFT_ELBOW][:2], landmarks[LEFT_WRIST][:2])
    right_angle = calculate_angle(landmarks[RIGHT_SHOULDER][:2], landmarks[RIGHT_ELBOW][:2], landmarks[RIGHT_WRIST][:2])

    ok_left = ELBOW_MIN_DEG <= left_angle <= ELBOW_MAX_DEG
    ok_right = ELBOW_MIN_DEG <= right_angle <= ELBOW_MAX_DEG

    if ok_left and ok_right:
        return 'GOOD', f"팔꿈치 각도 양호: L {int(left_angle)}°, R {int(right_angle)}°"

    # 어떤 쪽이든 벗어난 경우 메시지 생성
    issues = []
    if not ok_left:
        issues.append(f"왼쪽 {int(left_angle)}°")
    if not ok_right:
        issues.append(f"오른쪽 {int(right_angle)}°")
    return 'ELBOW_ANGLE_ISSUE', f"팔꿈치 각도 조정: {', '.join(issues)}. {ELBOW_MIN_DEG}°~{ELBOW_MAX_DEG}° 유지하세요."


def analyze_eye_level(landmarks):
    """
    시선/눈높이 분석: 모니터를 과도하게 내려다보는지 추정.
    - 입력: landmarks - MediaPipe Pose 33개 랜드마크 [x,y,z] 리스트.
    - 방법: 코와 양쪽 눈의 y좌표 차이를 계산하여 임계값보다 크면 'LOOKING_DOWN'.
    - 반환: (status, message). 'GOOD' 또는 'LOOKING_DOWN'.
    """
    # 코와 눈 위치로 고개 숙임 추정
    NOSE = 0
    LEFT_EYE, RIGHT_EYE = 2, 5

    nose_y = landmarks[NOSE][1]
    eye_y = (landmarks[LEFT_EYE][1] + landmarks[RIGHT_EYE][1]) / 2.0
    diff = nose_y - eye_y  # +면 코가 더 아래(고개 숙임)

    if diff > EYE_DOWN_NOSE_DIFF:
        return 'LOOKING_DOWN', "모니터를 너무 내려다봅니다. 눈높이에 맞추고 목의 긴장을 풀어주세요."
    return 'GOOD', "시선/눈높이 양호"