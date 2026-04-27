"""Gesture classifiers.

is_fist:   four non-thumb fingertips curled toward the wrist.
is_halt:   fist + the elbow bent close to a right angle (the conductor's
           halt sign — forearm vertical, fist up).
"""
import numpy as np

FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)

# MediaPipe Pose indices
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 13, 15
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 12, 14, 16


def is_fist(hand_pts):
    if hand_pts is None:
        return False
    wrist = hand_pts[0]
    closed = 0
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        d_tip = float(np.linalg.norm(hand_pts[tip] - wrist))
        d_pip = float(np.linalg.norm(hand_pts[pip] - wrist))
        # tip should not be meaningfully farther from the wrist than the
        # PIP joint when curled.
        if d_tip <= d_pip * 1.05:
            closed += 1
    return closed >= 3


def _arm_angle(pose_pts, shoulder_i, elbow_i, wrist_i, vis_threshold=0.5):
    if pose_pts is None:
        return None
    if (
        pose_pts[shoulder_i, 2] < vis_threshold
        or pose_pts[elbow_i, 2] < vis_threshold
        or pose_pts[wrist_i, 2] < vis_threshold
    ):
        return None
    a = pose_pts[shoulder_i, :2] - pose_pts[elbow_i, :2]
    b = pose_pts[wrist_i, :2] - pose_pts[elbow_i, :2]
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-3 or nb < 1e-3:
        return None
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def best_arm_angle(pose_pts):
    """Return (side, angle) for whichever arm is closer to 90°."""
    candidates = []
    left = _arm_angle(pose_pts, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    if left is not None:
        candidates.append(("left", left))
    right = _arm_angle(pose_pts, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    if right is not None:
        candidates.append(("right", right))
    if not candidates:
        return None, None
    return min(candidates, key=lambda x: abs(x[1] - 90.0))


def is_halt(hand_pts, pose_pts, angle_tol_deg=25.0):
    if not is_fist(hand_pts):
        return False
    _, ang = best_arm_angle(pose_pts)
    if ang is None:
        return False
    return abs(ang - 90.0) <= angle_tol_deg
