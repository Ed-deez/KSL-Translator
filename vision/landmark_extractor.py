
class LandmarkExtractor:

    @staticmethod
    def extract(result):

        # Check whether MediaPipe detected a hand.
        if not result.hand_landmarks:
            return None

        # Get the first detected hand.
        hand = result.hand_landmarks[0]

        # Store the coordinates of all 21 landmarks.
        landmarks = []

        for landmark in hand:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        return landmarks