import cv2
import mediapipe as mp
import numpy as np
import os


VIDEO_PATH = r"C:\Users\dalia\Downloads\val_rgb_front_clips\raw_videos\3EjKvwck6ss_0-1-rgb_front.mp4"

HAND_MODEL = r"models\hand_landmarker.task"
POSE_MODEL = r"models\pose_landmarker_full.task"
FACE_MODEL = r"models\face_landmarker.task"


# -------------------------
# HANDS
# -------------------------

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)


def extract_hand_features(result):

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    for hand_landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness
    ):

        values = []

        for lm in hand_landmarks:
            values.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        values = np.array(values, dtype=np.float32)

        hand_label = handedness[0].category_name

        if hand_label == "Left":
            left_hand = values

        elif hand_label == "Right":
            right_hand = values

    return np.concatenate([
        left_hand,
        right_hand
    ])


# -------------------------
# POSE
# -------------------------

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

pose_options = PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=POSE_MODEL
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)


def extract_pose_features(result):

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


# -------------------------
# FACE
# -------------------------

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

face_options = FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=FACE_MODEL
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)


def extract_face_features(result):

    face_values = np.zeros(52, dtype=np.float32)

    if result.face_blendshapes:

        face_values = np.array(
            [
                blendshape.score
                for blendshape in result.face_blendshapes[0]
            ],
            dtype=np.float32
        )

    return face_values


# -------------------------
# VIDEO
# -------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

sequence = []


with HandLandmarker.create_from_options(hand_options) as hand_detector, \
     PoseLandmarker.create_from_options(pose_options) as pose_detector, \
     FaceLandmarker.create_from_options(face_options) as face_detector:

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (frame_number / fps) * 1000
        )


        # Detect landmarks
        hand_result = hand_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        pose_result = pose_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        face_result = face_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # Extract features
        hand_features = extract_hand_features(
            hand_result
        )

        pose_features = extract_pose_features(
            pose_result
        )

        face_features = extract_face_features(
            face_result
        )


        # Combine everything for this frame
        frame_features = np.concatenate([
            hand_features,
            pose_features,
            face_features
        ])


        sequence.append(frame_features)

        frame_number += 1


cap.release()


# Convert entire video to NumPy sequence
sequence = np.array(
    sequence,
    dtype=np.float32
)


print("Final sequence shape:", sequence.shape)


# Save
os.makedirs(
    "data/processed",
    exist_ok=True
)

np.save(
    "data/processed/combined_sequence.npy",
    sequence
)

print("Sequence saved!")