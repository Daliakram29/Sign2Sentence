import cv2
import mediapipe as mp
VIDEO_PATH = r"C:\Users\dalia\Downloads\val_rgb_front_clips\raw_videos\3EjKvwck6ss_0-1-rgb_front.mp4"
MODEL_PATH = r"models\hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

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

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

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

cap.release()
cv2.destroyAllWindows()