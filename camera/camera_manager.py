import cv2


class CameraManager:

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.camera = None

    def start(self):

        print(f"Trying to open camera {self.camera_index}...")

        self.camera = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None

            raise RuntimeError(
                f"Could not open camera {self.camera_index}. "
                "Check Windows camera permissions and camera index."
            )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        width = int(
            self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        print(
            f"Camera {self.camera_index} opened successfully "
            f"at {width}x{height}."
        )

    def read(self):

        if self.camera is None:
            raise RuntimeError(
                "Camera has not been started."
            )

        success, frame = self.camera.read()

        if not success:
            raise RuntimeError(
                "Camera opened, but a frame could not be read."
            )

        return frame

    def stop(self):

        if self.camera is not None:
            self.camera.release()
            self.camera = None