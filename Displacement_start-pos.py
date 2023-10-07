import cv2 as cv
import mediapipe as mp
import numpy as np

def process_frame(frame):
    adjusted_midpoint = None
    if frame is None:
        return None, None

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        left_hip, right_hip, left_shoulder, right_shoulder = None, None, None, None

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if landmark.visibility > 0.5:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                if idx == mp_pose.PoseLandmark.LEFT_HIP.value:
                    left_hip = np.array([x, y])
                elif idx == mp_pose.PoseLandmark.RIGHT_HIP.value:
                    right_hip = np.array([x, y])
                elif idx == mp_pose.PoseLandmark.LEFT_SHOULDER.value:
                    left_shoulder = np.array([x, y])
                elif idx == mp_pose.PoseLandmark.RIGHT_SHOULDER.value:
                    right_shoulder = np.array([x, y])

        if left_hip is not None and right_hip is not None and left_shoulder is not None and right_shoulder is not None:
            hip_midpoint = (left_hip + right_hip) // 2
            shoulder_midpoint = (left_shoulder + right_shoulder) // 2
            adjusted_midpoint = (hip_midpoint + shoulder_midpoint) // 2
            cv.circle(frame, tuple(adjusted_midpoint), 5, (0, 255, 255), -1)

    return frame, adjusted_midpoint

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize the video file readers
video_path1 = 'video/deadpoint copy.mov'
video_path2 = 'video/deadpoint.mov'
cap1 = cv.VideoCapture(video_path1)
cap2 = cv.VideoCapture(video_path2)

midpoints1 = []
midpoints2 = []
# ... [rest of the imports and function definitions]

FONT_SCALE = 1.5
FONT_COLOR = (255, 255, 255)  # White
RECT_COLOR = (0, 0, 0)  # Black

while True:
    hasFrame1, frame1 = cap1.read()
    hasFrame2, frame2 = cap2.read()

    if not hasFrame1 and not hasFrame2:
        # Once both videos have finished playing, keep the window open until 'q' is pressed
        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                cap1.release()
                cap2.release()
                cv.destroyAllWindows()
                exit()

    frame1, adjusted_midpoint1 = process_frame(frame1)
    frame2, adjusted_midpoint2 = process_frame(frame2)

    if adjusted_midpoint1 is not None:
        midpoints1.append(adjusted_midpoint1)
        for i in range(1, len(midpoints1)):
            cv.line(frame1, tuple(midpoints1[i-1]), tuple(midpoints1[i]), (255, 105, 180), 4)
        displacement1 = np.linalg.norm(adjusted_midpoint1 - midpoints1[0])
        (text_width, text_height), _ = cv.getTextSize(f"Displacement: {displacement1:.2f} pixels", cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)
        cv.rectangle(frame1, (10, 10), (10 + text_width, 10 + text_height + 10), RECT_COLOR, -1)
        cv.putText(frame1, f"Displacement: {displacement1:.2f} pixels", (10, 10 + text_height), cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, 2)

    if adjusted_midpoint2 is not None:
        midpoints2.append(adjusted_midpoint2)
        for i in range(1, len(midpoints2)):
            cv.line(frame2, tuple(midpoints2[i-1]), tuple(midpoints2[i]), (255, 105, 180), 4)
        displacement2 = np.linalg.norm(adjusted_midpoint2 - midpoints2[0])
        (text_width, text_height), _ = cv.getTextSize(f"Displacement: {displacement2:.2f} pixels", cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)
        cv.rectangle(frame2, (10, 10), (10 + text_width, 10 + text_height + 10), RECT_COLOR, -1)
        cv.putText(frame2, f"Displacement: {displacement2:.2f} pixels", (10, 10 + text_height), cv.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, 2)

    combined_frame = np.hstack((frame1, frame2))
    cv.imshow('Side by Side Videos', combined_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()
