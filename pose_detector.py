"""
Pose detection utility using MediaPipe Pose.
- Provides a lightweight wrapper to detect and draw pose landmarks.
- Returns 33 landmarks as normalized [x, y, z] per frame when available.
Note: Designed for front-facing laptop cameras; depth(z) is approximate.
"""
import cv2
import mediapipe as mp

class PoseDetector:
    """
    MediaPipe Pose 래퍼 클래스.
    - find_pose: 입력 프레임에서 포즈를 추정하고, 옵션에 따라 랜드마크를 그립니다.
    - find_landmarks: 추정된 33개 랜드마크의 [x,y,z] 리스트를 반환합니다.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        PoseDetector 클래스의 생성자입니다. MediaPipe Pose 모델을 초기화합니다.
        - min_detection_confidence: 자세 감지가 성공한 것으로 간주되는 최소 신뢰도 값.
        - min_tracking_confidence: 랜드마크 추적이 성공한 것으로 간주되는 최소 신뢰도 값.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
        """
        입력 프레임에서 포즈를 추정하고, draw=True이면 랜드마크를 프레임에 그려 반환합니다.
        - 입력: img (BGR ndarray), draw (bool)
        - 반환: 랜드마크가 그려진(또는 원본) BGR 이미지
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MediaPipe는 RGB 이미지를 사용하므로 변환합니다.
        self.results = self.pose.process(img_rgb)  # 자세 추정을 수행합니다.
        if self.results.pose_landmarks and draw:
            # 감지된 랜드마크가 있고 draw 옵션이 True이면, 프레임에 랜드마크와 연결선을 그립니다.
            self.mp_drawing.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_landmarks(self, img):
        """
        감지된 33개의 랜드마크 좌표 [x, y, z]를 리스트로 반환합니다.
        - 입력: img (랜드마크가 감지된 이미지 프레임, 현재 구현에서는 사용되지 않음)
        - 반환: 랜드마크 좌표 리스트. 랜드마크가 없으면 빈 리스트를 반환.
        """
        landmark_list = []
        if self.results.pose_landmarks:
            for lm in self.results.pose_landmarks.landmark:
                # 각 랜드마크의 정규화된 x, y, z 좌표를 리스트에 추가합니다.
                landmark_list.append([lm.x, lm.y, lm.z])
        return landmark_list