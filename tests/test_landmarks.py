import cv2

from camera.camera_manager import CameraManager
from vision.hand_detector import HandDetector
from vision.landmark_extractor import LandmarkExtractor
from vision.feature_normalizer import FeatureNormalizer


MODEL_PATH = "assets/hand_landmarker.task"


def main():

    camera = CameraManager(0)
    detector = HandDetector(MODEL_PATH)

    camera.start()

    print("MediaPipe hand detection started.")
    print("Press Q to quit.")

    try:
        while True:

            frame = camera.read()

            result = detector.detect(frame)

            landmarks = LandmarkExtractor.extract(result)

            if landmarks is not None:

                normalized = FeatureNormalizer.normalize(
            landmarks
            )

                print(
                    f"Raw landmarks: {len(landmarks)}"
                )

                print(
                    f"Normalized features: {len(normalized)}"
                )

            if landmarks is not None:

                print(
                    f"Hand detected - "
                    f"{len(landmarks)} values"
                )

                cv2.putText(
                    frame,
                    "HAND DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "NO HAND",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "KSL Translator - Hand Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.stop()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()