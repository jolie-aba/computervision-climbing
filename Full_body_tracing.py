import cv2 as cv
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize the video file reader
video_path = 'video/IMG_0521.mov'
cap = cv.VideoCapture(video_path)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    # Convert the BGR image to RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw the pose landmarks on the frame
    if results.pose_landmarks:
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if results.pose_landmarks.landmark[start_idx].visibility > 0.5 and results.pose_landmarks.landmark[end_idx].visibility > 0.5:
                start = tuple([int(results.pose_landmarks.landmark[start_idx].x * frame.shape[1]), int(results.pose_landmarks.landmark[start_idx].y * frame.shape[0])])
                end = tuple([int(results.pose_landmarks.landmark[end_idx].x * frame.shape[1]), int(results.pose_landmarks.landmark[end_idx].y * frame.shape[0])])
                cv.line(frame, start, end, (0, 255, 0), 3)
                cv.ellipse(frame, start, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, end, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    cv.imshow('Pose Estimation using MediaPipe', frame)

cap.release()
cv.destroyAllWindows()
