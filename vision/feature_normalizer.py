import numpy as np


class FeatureNormalizer:

    @staticmethod
    def normalize(landmarks):

        points = np.array(
            landmarks,
            dtype=np.float32
        ).reshape(21, 3)

        # Wrist = landmark 0
        wrist = points[0].copy()

        # Move the wrist to the origin
        points = points - wrist

        # Scale based on maximum distance
        scale = np.max(np.abs(points))

        if scale > 0:
            points = points / scale

        return points.flatten()