import cv2
import mediapipe as mp
import numpy as np

VIDEO_PATH = r"C:\Users\dalia\Downloads\val_rgb_front_clips\raw_videos\3EjKvwck6ss_0-1-rgb_front.mp4"
MODEL_PATH = r"models\pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
sequence = []

connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

def extract_pose_features(result):
    # 33 landmarks × 3 coordinates = 99 values
    pose_values = np.zeros(99, dtype=np.float32)

    if result.pose_landmarks:
        values = []

        for lm in result.pose_landmarks[0]:
            values.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        pose_values = np.array(values, dtype=np.float32)

    return pose_values

with PoseLandmarker.create_from_options(options) as detector:
    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int((frame_number / fps) * 1000)

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        frame_features = extract_pose_features(result)
        sequence.append(frame_features)

        h, w, _ = frame.shape

        for pose_landmarks in result.pose_landmarks:
            points = []

            for lm in pose_landmarks:
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                points.append((cx, cy))

                cv2.circle(
                    frame,
                    (cx, cy),
                    4,
                    (0, 255, 0),
                    -1
                )

            for connection in connections:
                start = connection.start
                end = connection.end

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 255, 255),
                    2
                )

        cv2.imshow("How2Sign - Pose Landmarks", frame)

        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

sequence = np.array(sequence, dtype=np.float32)

print("Pose sequence shape:", sequence.shape)

cap.release()
cv2.destroyAllWindows()