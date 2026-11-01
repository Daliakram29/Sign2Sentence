import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import os


# -------------------------
# PATHS
# -------------------------

VIDEO_FOLDER = Path(
    r"C:\Users\dalia\Downloads\val_rgb_front_clips\raw_videos"
)

CSV_PATH = r"C:\Users\dalia\Downloads\how2sign_realigned_val.csv"

HAND_MODEL = r"models\hand_landmarker.task"
POSE_MODEL = r"models\pose_landmarker_full.task"
FACE_MODEL = r"models\face_landmarker.task"

OUTPUT_FOLDER = Path("data/processed")

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------
# LOAD ENGLISH SENTENCES
# -------------------------

df = pd.read_csv(
    CSV_PATH,
    sep="\t"
)


# -------------------------
# MEDIAPIPE SETUP
# -------------------------

HandLandmarker = mp.tasks.vision.HandLandmarker
PoseLandmarker = mp.tasks.vision.PoseLandmarker
FaceLandmarker = mp.tasks.vision.FaceLandmarker

RunningMode = mp.tasks.vision.RunningMode


hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)


pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=POSE_MODEL
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)


face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=FACE_MODEL
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)


# -------------------------
# FEATURE FUNCTIONS
# -------------------------

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

        values = np.array(
            values,
            dtype=np.float32
        )

        hand_label = handedness[0].category_name

        if hand_label == "Left":
            left_hand = values

        elif hand_label == "Right":
            right_hand = values

    return np.concatenate([
        left_hand,
        right_hand
    ])


def extract_pose_features(result):

    pose_values = np.zeros(
        99,
        dtype=np.float32
    )

    if result.pose_landmarks:

        values = []

        for lm in result.pose_landmarks[0]:
            values.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        pose_values = np.array(
            values,
            dtype=np.float32
        )

    return pose_values


def extract_face_features(result):

    face_values = np.zeros(
        52,
        dtype=np.float32
    )

    if result.face_blendshapes:

        face_values = np.array(
            [
                blendshape.score
                for blendshape
                in result.face_blendshapes[0]
            ],
            dtype=np.float32
        )

    return face_values


# -------------------------
# PROCESS ONE VIDEO
# -------------------------

def process_video(video_path):

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    sequence = []
    frame_number = 0

    with HandLandmarker.create_from_options(hand_options) as hand_detector, \
         PoseLandmarker.create_from_options(pose_options) as pose_detector, \
         FaceLandmarker.create_from_options(face_options) as face_detector:

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

            hand_features = extract_hand_features(
                hand_result
            )

            pose_features = extract_pose_features(
                pose_result
            )

            face_features = extract_face_features(
                face_result
            )

            frame_features = np.concatenate([
                hand_features,
                pose_features,
                face_features
            ])

            sequence.append(frame_features)

            frame_number += 1

    cap.release()

    return np.array(
        sequence,
        dtype=np.float32
    )


# -------------------------
# PROCESS DATASET
# -------------------------

video_files = list(
    VIDEO_FOLDER.glob("*.mp4")
)

# Only 10 videos for our first test
video_files = video_files[:10]

metadata = []


with HandLandmarker.create_from_options(hand_options) as hand_detector, \
     PoseLandmarker.create_from_options(pose_options) as pose_detector, \
     FaceLandmarker.create_from_options(face_options) as face_detector:
    
    for video_path in video_files:

        video_name = video_path.stem

        row = df[
            df["SENTENCE_NAME"] == video_name
            ]

        if row.empty:
            continue

        sentence = row.iloc[0]["SENTENCE"]

        print(f"Processing: {video_name}")

        sequence = process_video(video_path)

        output_path = (
                OUTPUT_FOLDER /
                f"{video_name}.npy"
        )

        np.save(
            output_path,
            sequence
        )

        metadata.append({
            "video_name": video_name,
            "sequence_file": str(output_path),
            "frames": sequence.shape[0],
            "features": sequence.shape[1],
            "sentence": sentence
        })


# -------------------------
# SAVE METADATA
# -------------------------

metadata_df = pd.DataFrame(
    metadata
)

metadata_df.to_csv(
    "data/metadata.csv",
    index=False
)

print(
    f"\nFinished processing {len(metadata_df)} videos."
)