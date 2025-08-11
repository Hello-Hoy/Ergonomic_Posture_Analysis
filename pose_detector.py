import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
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