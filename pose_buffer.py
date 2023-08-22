class PoseBuffer:
    def __init__(self, max_size=100):
        self.buffer = []  # List of (timestamp, pose_landmarks) pairs
        self.max_size = max_size

    def add_pose(self, timestamp, pose_landmarks):
        self.buffer.append((timestamp, pose_landmarks))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_latest_pose(self):
        if self.buffer:
            return self.buffer[-1]  # Return the latest pose_landmarks
        return None
