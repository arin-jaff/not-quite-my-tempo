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
):
    h, w = frame.shape[:2]

    # Top-left: status panel
    _panel(frame, (10, 10), (370, 138))
    status_color = (
        GREEN if status == "PLAYING"
        else RED if status == "HALTED"
        else YELLOW
    )
    _shadowed(frame, f"STATUS: {status}", (22, 40), 0.7, status_color)
    _shadowed(frame, f"GESTURE: {classification}", (22, 68), 0.6, WHITE)
    _shadowed(frame, f"SONG: {_truncate(song_name, 32)}", (22, 94), 0.5, DIM, 1)
    _shadowed(frame, f"TARGET: {target_bpm:5.1f} BPM", (22, 122), 0.55, WHITE, 1)

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
