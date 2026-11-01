import cv2
import mediapipe as mp
import numpy as np
import os

#paths
VIDEO_PATH = r"C:\Users\dalia\Downloads\val_rgb_front_clips\raw_videos\3EjKvwck6ss_0-1-rgb_front.mp4"
MODEL_PATH = r"models\hand_landmarker.task"

#mediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

sequence = []
#define the function
def extract_hand_features(result):
    # 21 landmarks × 3 coordinates = 63 values
    #we always reserve space for both hands
    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    for hand_landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness
    ):
        hand_values = []

        for lm in hand_landmarks:
            hand_values.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        hand_values = np.array(
            hand_values,
            dtype=np.float32
        )

        hand_label = handedness[0].category_name

        if hand_label == "Left":
            left_hand = hand_values

        elif hand_label == "Right":
            right_hand = hand_values

    # Left first, then Right
    return np.concatenate([
        left_hand,
        right_hand
    ])

#open video:
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


#detector context
#create a list for the whole video
with HandLandmarker.create_from_options(options) as detector:
    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV uses BGR, MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Time of this frame in milliseconds
        timestamp_ms = int((frame_number / fps) * 1000)
#extract features inside the loop
        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        frame_features = extract_hand_features(result)
        sequence.append(frame_features)

        h, w, _ = frame.shape

        for hand_landmarks in result.hand_landmarks:

            points = []

            # Convert normalized landmarks to pixel coordinates
            for lm in hand_landmarks:
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

            # Connect the landmark points
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

        # Show processed video
        cv2.imshow("How2Sign - Hand Landmarks", frame)

        # Move to the next frame number
        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

sequence = np.array(sequence, dtype=np.float32)

print("Sequence shape:", sequence.shape)
print("First frame features: ")
print(sequence[0])

print("Second frame features: ")
print(sequence[0].shape)

print("First 10 values: ")
print(sequence[0][:10])
print("Non-zero values in first frame:")
print(np.count_nonzero(sequence[0]))
cap.release()
cv2.destroyAllWindows()

#save the sequence
os.makedirs("data/processed", exist_ok=True)

np.save(
    "data/processed/test_sequence.npy",
    sequence
)

print("Sequence saved!")