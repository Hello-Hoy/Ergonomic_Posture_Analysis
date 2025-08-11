# main.py
import cv2
import time
from pose_detector import PoseDetector
from ergonomics_rules import analyze_head_posture, analyze_shoulder_posture
from feedback_handler import FeedbackHandler

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    feedback = FeedbackHandler()

    last_posture_issue = None
    issue_start_time = None
    ISSUE_THRESHOLD_SECONDS = 3.0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = detector.find_pose(img, draw=True)
        landmarks = detector.find_landmarks(img)

        current_issue = None
        if landmarks:
            # 자세 분석
            head_status, head_msg = analyze_head_posture(landmarks)
            shoulder_status, shoulder_msg = analyze_shoulder_posture(landmarks)

            if head_status!= 'GOOD':
                current_issue = head_msg
            elif shoulder_status!= 'GOOD':
                current_issue = shoulder_msg
            else:
                current_issue = "좋은 자세입니다."

        # 피드백 로직
        if current_issue and "좋은 자세입니다." not in current_issue:
            if current_issue!= last_posture_issue:
                last_posture_issue = current_issue
                issue_start_time = time.time()
            elif time.time() - issue_start_time > ISSUE_THRESHOLD_SECONDS:
                feedback.provide_text_feedback(current_issue)
                feedback.provide_voice_feedback(current_issue)
                issue_start_time = time.time() # 피드백 후 타이머 리셋
        else:
            last_posture_issue = None
            issue_start_time = None

        cv2.imshow("Ergonomic Workspace Analyzer", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()