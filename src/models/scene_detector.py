class SceneDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def detect_scenes(self, frames):
        scene_changes = []
        previous_frame = None

        for i, frame in enumerate(frames):
            if previous_frame is not None:
                change_score = self._calculate_change(previous_frame, frame)
                if change_score > self.threshold:
                    scene_changes.append(i)
            previous_frame = frame

        return scene_changes

    def _calculate_change(self, frame1, frame2):
        # Placeholder for actual change calculation logic
        return abs(frame1.mean() - frame2.mean())  # Example using mean pixel value difference