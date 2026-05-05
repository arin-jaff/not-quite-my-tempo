"""HUD overlay: status, BPM, off-tempo banner, history graph, toasts."""
import time

import cv2
import numpy as np

WHITE = (255, 255, 255)
RED = (40, 40, 230)
GREEN = (90, 220, 90)
YELLOW = (60, 220, 240)
DIM = (180, 180, 180)
BLACK = (0, 0, 0)
GRAPH_LINE = (200, 200, 70)
PEAK_LINE = (60, 60, 220)
TOAST_TEXT = (255, 235, 200)
HAND_BONE = (220, 200, 90)
HAND_JOINT = (240, 240, 240)
ARM_BONE = (220, 140, 80)

# MediaPipe Hands topology — 21 landmarks
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)
FINGER_TIPS = (4, 8, 12, 16, 20)
WRIST_IDX = 0


def _shadowed(img, text, org, scale=0.7, color=WHITE, thickness=2):
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _panel(frame, p0, p1, alpha=0.45):
    overlay = frame.copy()
    cv2.rectangle(overlay, p0, p1, BLACK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, p0, p1, DIM, 1)


def draw_hud(
    frame,
    *,
    status,
    classification,
    current_bpm,
    target_bpm,
    on_tempo,
    history,
    peaks,
    events,
    song_name,
    hand_pts=None,
    pose_pts=None,
    is_fist_flag=False,
    is_halt_flag=False,
    arm_side=None,
    arm_angle=None,
    rate=1.0,
):
    h, w = frame.shape[:2]

    # Skeleton goes underneath the panels so text stays readable
    _draw_skeleton(
        frame, hand_pts, pose_pts,
        is_fist_flag=is_fist_flag,
        is_halt_flag=is_halt_flag,
        arm_side=arm_side,
        arm_angle=arm_angle,
    )

    # Top-left: status panel
    _panel(frame, (10, 10), (370, 162))
    status_color = (
        GREEN if status == "PLAYING"
        else RED if status == "HALTED"
        else YELLOW
    )
    _shadowed(frame, f"STATUS: {status}", (22, 40), 0.7, status_color)
    _shadowed(frame, f"GESTURE: {classification}", (22, 68), 0.6, WHITE)
    _shadowed(frame, f"SONG: {_truncate(song_name, 32)}", (22, 94), 0.5, DIM, 1)
    _shadowed(frame, f"TARGET: {target_bpm:5.1f} BPM", (22, 122), 0.55, WHITE, 1)
    rate_color = GREEN if abs(rate - 1.0) < 0.02 else YELLOW
    _shadowed(frame, f"RATE:   {rate:4.2f}x", (22, 148), 0.55, rate_color, 1)

    # Top-right: current BPM readout
    _panel(frame, (w - 270, 10), (w - 10, 100))
    _shadowed(frame, "CURRENT BPM", (w - 258, 38), 0.55, DIM, 1)
    cur_text = f"{current_bpm:5.1f}" if current_bpm is not None else "  -- "
    bpm_color = GREEN if on_tempo else (RED if current_bpm is not None else DIM)
    _shadowed(frame, cur_text, (w - 258, 82), 1.2, bpm_color, 2)

    # Off-tempo banner
    if current_bpm is not None and not on_tempo:
        _draw_banner(frame, "NOT QUITE MY TEMPO", RED)

    # History graph at bottom
    _draw_graph(frame, history, peaks)

    # Event toaster (above graph)
    _draw_events(frame, events)


def _truncate(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


def _draw_banner(frame, text, color):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)
    x = (w - tw) // 2
    y = 60
    cv2.rectangle(frame, (x - 18, y - th - 14), (x + tw + 18, y + 14),
                  BLACK, -1)
    cv2.rectangle(frame, (x - 18, y - th - 14), (x + tw + 18, y + 14),
                  color, 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 3,
                cv2.LINE_AA)


def _draw_graph(frame, history, peaks):
    h, w = frame.shape[:2]
    gx0, gy0 = 10, h - 130
    gx1, gy1 = w - 10, h - 20
    _panel(frame, (gx0, gy0), (gx1, gy1), alpha=0.5)
    _shadowed(frame, "WRIST Y / TIME", (gx0 + 8, gy0 + 18), 0.45, DIM, 1)

    if not history or len(history) < 2:
        return

    ts = np.array([t for t, _ in history], dtype=np.float64)
    ys = np.array([y for _, y in history], dtype=np.float64)
    t_min, t_max = float(ts.min()), float(ts.max())
    span = max(t_max - t_min, 1e-3)
    y_min, y_max = float(ys.min()), float(ys.max())
    y_span = max(y_max - y_min, 1.0)

    inner_x0, inner_x1 = gx0 + 8, gx1 - 8
    inner_y0, inner_y1 = gy0 + 26, gy1 - 8

    pts = []
    for t, y in zip(ts, ys):
        px = int(inner_x0 + (t - t_min) / span * (inner_x1 - inner_x0))
        py = int(inner_y0 + (y - y_min) / y_span * (inner_y1 - inner_y0))
        pts.append((px, py))
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], GRAPH_LINE, 2, cv2.LINE_AA)

    for tp in peaks:
        if t_min <= tp <= t_max:
            px = int(inner_x0 + (tp - t_min) / span * (inner_x1 - inner_x0))
            cv2.line(frame, (px, inner_y0), (px, inner_y1), PEAK_LINE, 1)


def _draw_skeleton(
    frame, hand_pts, pose_pts, *,
    is_fist_flag, is_halt_flag, arm_side, arm_angle,
):
    # Hand bones + joints
    if hand_pts is not None:
        bone_color = RED if is_halt_flag else (GREEN if is_fist_flag else HAND_BONE)
        for a, b in HAND_CONNECTIONS:
            pa = (int(hand_pts[a, 0]), int(hand_pts[a, 1]))
            pb = (int(hand_pts[b, 0]), int(hand_pts[b, 1]))
            cv2.line(frame, pa, pb, bone_color, 2, cv2.LINE_AA)
        for i in range(hand_pts.shape[0]):
            p = (int(hand_pts[i, 0]), int(hand_pts[i, 1]))
            r = 5 if i in FINGER_TIPS else 3
            cv2.circle(frame, p, r, HAND_JOINT, -1, cv2.LINE_AA)
        # Wrist marker
        wp = (int(hand_pts[WRIST_IDX, 0]), int(hand_pts[WRIST_IDX, 1]))
        cv2.circle(frame, wp, 9, YELLOW, 2, cv2.LINE_AA)
        _shadowed(frame, "WRIST", (wp[0] + 12, wp[1] - 8), 0.45, YELLOW, 1)
        if is_halt_flag:
            _shadowed(frame, "HALT", (wp[0] + 12, wp[1] + 14), 0.55, RED, 2)
        elif is_fist_flag:
            _shadowed(frame, "FIST", (wp[0] + 12, wp[1] + 14), 0.5, GREEN, 1)

    # Active arm: shoulder → elbow → wrist
    if pose_pts is not None and arm_side in ("left", "right"):
        if arm_side == "left":
            s_i, e_i, w_i = 11, 13, 15
        else:
            s_i, e_i, w_i = 12, 14, 16
        if (
            pose_pts[s_i, 2] > 0.3
            and pose_pts[e_i, 2] > 0.3
            and pose_pts[w_i, 2] > 0.3
        ):
            sp = (int(pose_pts[s_i, 0]), int(pose_pts[s_i, 1]))
            ep = (int(pose_pts[e_i, 0]), int(pose_pts[e_i, 1]))
            wp = (int(pose_pts[w_i, 0]), int(pose_pts[w_i, 1]))
            cv2.line(frame, sp, ep, ARM_BONE, 3, cv2.LINE_AA)
            cv2.line(frame, ep, wp, ARM_BONE, 3, cv2.LINE_AA)
            for p in (sp, ep, wp):
                cv2.circle(frame, p, 5, HAND_JOINT, -1, cv2.LINE_AA)
            if arm_angle is not None:
                near_right = abs(arm_angle - 90.0) <= 25.0
                color = GREEN if near_right else DIM
                _shadowed(
                    frame, f"{arm_angle:5.1f}°",
                    (ep[0] + 10, ep[1] + 6), 0.55, color, 1,
                )


def _draw_events(frame, events, ttl=3.5):
    h, w = frame.shape[:2]
    y = h - 150
    now = time.time()
    for t, msg in reversed(events):
        age = now - t
        alpha = max(0.0, 1.0 - age / ttl)
        if alpha <= 0:
            continue
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        x = 14
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 8, y - th - 8), (x + tw + 12, y + 8),
                      BLACK, -1)
        a = 0.55 * alpha + 0.05
        cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)
        cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    TOAST_TEXT, 1, cv2.LINE_AA)
        y -= th + 16
