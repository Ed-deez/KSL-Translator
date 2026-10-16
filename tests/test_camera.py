import cv2

from camera.camera_manager import CameraManager


def main():
    camera = CameraManager(camera_index=0)

    try:
        camera.start()

        print("Camera started.")
        print("Press Q to close.")

        while True:
            frame = camera.read()

            cv2.imshow("KSL Translator - Camera Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()