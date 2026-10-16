import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, model_path):
        self.model_path = model_path

        self.base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = (
            mp.tasks.vision.HandLandmarker
            .create_from_options(self.options)
        )

    def detect(self, frame):
        # OpenCV uses BGR.
        # MediaPipe expects an RGB image.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.landmarker.detect(mp_image)

        return result

    def close(self):
        self.landmarker.close()