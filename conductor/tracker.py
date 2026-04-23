"""MediaPipe Hands + Pose wrapper.

Returns landmarks in pixel coordinates so downstream modules don't need to
know about the frame size. Tracks one hand and one body — the conductor.
"""
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


class TrackingResult:
    def __init__(self, hand_pts, pose_pts, frame_size):
        self.hand_pts = hand_pts        # (21, 2) pixel xy, or None
        self.pose_pts = pose_pts        # (33, 3) pixel xy + visibility, or None
        self.frame_size = frame_size    # (w, h)

    @property
    def has_hand(self):
        return self.hand_pts is not None

    @property
    def has_pose(self):
        return self.pose_pts is not None

    @property
    def wrist_xy(self):
        if self.hand_pts is None:
            return None
        return float(self.hand_pts[0, 0]), float(self.hand_pts[0, 1])


class Tracker:
    def __init__(self):
        self._hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

    def process(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        hand_res = self._hands.process(frame_rgb)
        pose_res = self._pose.process(frame_rgb)

        hand_pts = None
        if hand_res.multi_hand_landmarks:
            lm = hand_res.multi_hand_landmarks[0].landmark
            hand_pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        pose_pts = None
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            pose_pts = np.array(
                [[p.x * w, p.y * h, p.visibility] for p in lm],
                dtype=np.float32,
            )

        return TrackingResult(hand_pts, pose_pts, (w, h))

    def close(self):
        self._hands.close()
        self._pose.close()
