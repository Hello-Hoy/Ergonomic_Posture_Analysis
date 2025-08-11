"""
PyCharm 프로젝트 설정 및 라이브러리 설치 안내
1) 새 프로젝트 생성 -> Python 인터프리터(가상환경) 선택/생성
2) 터미널에서 필수 라이브러리 설치:
   pip install opencv-python mediapipe numpy gTTS playsound pillow pygame
   - 음성 피드백은 gTTS + pygame mixer를 사용합니다.
3) 실행: python main.py (또는 PyCharm에서 Run)
팁:
- q 키를 누르면 종료됩니다.
- MediaPipe Pose로 33개 랜드마크를 검출하여 화면에 시각화합니다.
- 계산된 각도를 기준 임계값과 비교하여 텍스트/음성 피드백을 제공합니다.
"""
import cv2
import time
import os
import warnings
import numpy as np
from typing import List, Tuple

# Protobuf/3rd-party deprecation warnings can spam the console; hide them.
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Filter specific noisy protobuf UserWarning: 'SymbolDatabase.GetPrototype() is deprecated.'
warnings.filterwarnings("ignore", message=".*GetPrototype\(\) is deprecated.*")

# PIL (Pillow) for proper Korean text rendering on frames
try:
    from PIL import ImageFont, ImageDraw, Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# 다른 모듈에서 클래스와 함수를 가져옵니다.
from pose_detector import PoseDetector  # 자세 감지 모듈
from ergonomics_rules import (  # 인체공학 규칙 분석 모듈
    analyze_head_posture,
    analyze_shoulder_posture,
    analyze_elbow_posture,
    analyze_eye_level,
)
from feedback_handler import FeedbackHandler  # 피드백 처리 모듈

# ---- Text rendering helpers (support Korean) ----
_warned_text_fallback = False  # 한글 폰트 경고 메시지가 한 번만 출력되도록 하는 플래그


def _find_korean_font() -> str | None:
    """
    시스템에서 한글 렌더링이 가능한 폰트를 탐색해 경로를 반환합니다.
    - macOS/Windows/Linux의 대표 경로를 순회합니다.
    - 찾지 못하면 None을 반환합니다.
    """
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS default
        "/Library/Fonts/AppleSDGothicNeo.ttc",
        os.path.expanduser("~/Library/Fonts/AppleSDGothicNeo.ttc"),
        "/Library/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/Supplemental/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
        "C:/Windows/Fonts/malgun.ttf",  # Windows
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

_KR_FONT_PATH = _find_korean_font()  # 시스템에서 한글 폰트 경로를 찾습니다.


def draw_text_multiline(
    img: np.ndarray,
    lines: List[Tuple[str, Tuple[int, int, int]]],
    org: Tuple[int, int] = (10, 30),
    font_size: int = 24,
    line_gap: int = 6,
) -> None:
    """
    프레임에 여러 줄의 텍스트를 그립니다.
    lines의 각 항목은 (text, bgr_color) 형식입니다. PIL을 사용하여 한글을 올바르게 렌더링합니다.
    """
    global _warned_text_fallback

    if PIL_AVAILABLE and _KR_FONT_PATH:
        # 이미지를 BGR에서 RGB로 변환하고 PIL 이미지로 만듭니다.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(_KR_FONT_PATH, font_size)
        except Exception:
            font = ImageFont.load_default()  # 폰트 로드 실패 시 기본 폰트 사용

        x, y = org
        for text, bgr in lines:
            color_rgb = (bgr[2], bgr[1], bgr[0])  # BGR을 RGB로 변환
            draw.text((x, y), text, fill=color_rgb, font=font)
            # 텍스트의 높이를 측정하여 다음 줄의 y 좌표를 계산합니다.
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                h = bbox[3] - bbox[1]
            except Exception:
                h = font_size + 4
            y += h + line_gap

        # 이미지를 다시 OpenCV BGR 형식으로 변환합니다.
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        # PIL을 사용할 수 없으면 OpenCV의 기본 폰트를 사용합니다 (한글 깨짐 발생 가능).
        if not _warned_text_fallback:
            print("[안내] 한글 텍스트가 물음표(????)로 보이면 'pip install pillow' 설치 및 한글 폰트(Apple SD Gothic 등)를 시스템에 설치해 주세요.")
            _warned_text_fallback = True
        x, y = org
        for text, bgr in lines:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
            y += int(font_size * 1.4) + line_gap

def main():
    """
    메인 루프: 웹캠 프레임에서 포즈를 추정하고 4가지 항목(목, 어깨, 팔꿈치, 시선)을 분석해 피드백 제공합니다.
    노트북 전면 카메라 한계로 등/허리 평가는 제외됩니다. 'q' 키로 종료.
    """
    cap = cv2.VideoCapture(0)  # 웹캠을 엽니다 (0은 기본 카메라).
    detector = PoseDetector()  # 자세 감지기 객체를 생성합니다.
    feedback = FeedbackHandler()  # 피드백 처리기 객체를 생성합니다.

    last_posture_issue = None  # 마지막으로 감지된 자세 문제
    issue_start_time = None  # 문제가 시작된 시간
    ISSUE_THRESHOLD_SECONDS = 2.0  # 동일한 문제가 지속될 때 피드백을 제공하기 위한 시간 임계값 (3초)

    while cap.isOpened():  # 웹캠이 열려있는 동안 루프를 실행합니다.
        success, img = cap.read()  # 웹캠에서 프레임을 읽습니다.
        if not success:
            break

        img = cv2.flip(img, 1)  # 프레임을 좌우로 반전시켜 거울 모드로 만듭니다.
        img = detector.find_pose(img, draw=True)  # 프레임에서 자세를 감지하고 랜드마크를 그립니다.
        landmarks = detector.find_landmarks(img)  # 감지된 랜드마크 좌표를 가져옵니다.

        current_issue = None  # 현재 프레임에서 감지된 문제
        overlay_lines = []  # 화면에 표시할 텍스트 라인
        if landmarks:
            # 자세 분석: 4가지 항목을 평가합니다.
            head_status, head_msg = analyze_head_posture(landmarks)
            shoulder_status, shoulder_msg = analyze_shoulder_posture(landmarks)
            eye_status, eye_msg = analyze_eye_level(landmarks)
            elbow_status, elbow_msg = analyze_elbow_posture(landmarks)

            issues = [
                (head_status, head_msg),
                (shoulder_status, shoulder_msg),
                (eye_status, eye_msg),
                (elbow_status, elbow_msg),
            ]

            # 음성/1순위 피드백용 첫 번째 문제를 선택합니다.
            for status, msg in issues:
                if status != 'GOOD':
                    current_issue = msg
                    break

            if current_issue is None:
                # 모든 상태가 양호한 경우
                overlay_lines = [("좋은 자세입니다.", (0, 200, 0))]
            else:
                # 화면에는 상위 3개의 이슈를 동시에 표시합니다.
                for status, msg in issues:
                    if status != 'GOOD':
                        overlay_lines.append((msg, (0, 0, 255)))
                        if len(overlay_lines) >= 3:
                            break

        # 화면에 텍스트 피드백을 표시합니다.
        if overlay_lines:
            draw_text_multiline(img, overlay_lines, org=(10, 30), font_size=40, line_gap=8)

        # 피드백 로직
        if current_issue and "좋은 자세입니다." not in current_issue:
            if current_issue != last_posture_issue:
                # 새로운 문제가 감지되면, 문제와 시작 시간을 기록합니다.
                last_posture_issue = current_issue
                issue_start_time = time.time()
            elif time.time() - issue_start_time > ISSUE_THRESHOLD_SECONDS:
                # 동일한 문제가 3초 이상 지속되면 텍스트 및 음성 피드백을 제공합니다.
                feedback.provide_text_feedback(current_issue)
                feedback.provide_voice_feedback(current_issue)
                issue_start_time = time.time()  # 피드백 후 타이머를 리셋합니다.
        else:
            # 문제가 없으면 상태를 초기화합니다.
            last_posture_issue = None
            issue_start_time = None

        cv2.imshow("Ergonomic Workspace Analyzer", img)  # 화면에 이미지를 표시합니다.

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 루프를 종료합니다.
            break

    cap.release()  # 웹캠을 해제합니다.
    cv2.destroyAllWindows()  # 모든 창을 닫습니다.

if __name__ == "__main__":
    main()