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
        # Annotate hands, feet, head, and hips
        landmark_positions = {
            'Right wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
            'Left wrist': mp_pose.PoseLandmark.LEFT_WRIST,
            'Right heel': mp_pose.PoseLandmark.RIGHT_HEEL,
            'Left heel': mp_pose.PoseLandmark.LEFT_HEEL,
            'Nose': mp_pose.PoseLandmark.NOSE,
            'Right hip': mp_pose.PoseLandmark.RIGHT_HIP,
            'Left hip': mp_pose.PoseLandmark.LEFT_HIP,
        }
        
        for label, landmark in landmark_positions.items():
            landmark_point = results.pose_landmarks.landmark[landmark]
            if landmark_point.visibility > 0.5:
                landmark_px = (int(landmark_point.x * frame.shape[1]), int(landmark_point.y * frame.shape[0]))
                cv.circle(frame, landmark_px, 5, (255, 0, 0), -1)
                cv.putText(frame, f"{label}: {landmark_px}", (landmark_px[0] + 10, landmark_px[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    cv.imshow('Pose Estimation using MediaPipe', frame)

cap.release()
cv.destroyAllWindows()
