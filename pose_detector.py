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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_landmarks(self, img):
        landmark_list = []
        if self.results.pose_landmarks:
            for lm in self.results.pose_landmarks.landmark:
                # 3D 좌표를 저장합니다. z 좌표는 깊이를 나타냅니다.
                landmark_list.append([lm.x, lm.y, lm.z])
        return landmark_list