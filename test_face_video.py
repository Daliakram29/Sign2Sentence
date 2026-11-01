import cv2
import mediapipe as mp
import numpy as np

VIDEO_PATH = r"C:\Users\dalia\Downloads\val_rgb_front_clips\raw_videos\3EjKvwck6ss_0-1-rgb_front.mp4"
MODEL_PATH = r"models\face_landmarker.task"


# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)


# Extract numerical face features
def extract_face_features(result):

    # 478 landmarks × 3 coordinates = 1434 features
    face_values = np.zeros(478 * 3, dtype=np.float32)

    if result.face_landmarks:

        values = []

        for lm in result.face_landmarks[0]:
            values.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        face_values = np.array(
            values,
            dtype=np.float32
        )

    return face_values


# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

sequence = []


with FaceLandmarker.create_from_options(options) as detector:

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # OpenCV BGR → RGB
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


        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # Extract numerical features
        frame_features = extract_face_features(result)

        sequence.append(frame_features)


        # Draw facial landmarks
        h, w, _ = frame.shape

        for face_landmarks in result.face_landmarks:

            for lm in face_landmarks:

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(
                    frame,
                    (cx, cy),
                    1,
                    (0, 255, 0),
                    -1
                )


        cv2.imshow(
            "How2Sign - Face Landmarks",
            frame
        )


        frame_number += 1


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# Convert sequence into NumPy array
sequence = np.array(
    sequence,
    dtype=np.float32
)


print(
    "Face sequence shape:",
    sequence.shape
)


cap.release()
cv2.destroyAllWindows()