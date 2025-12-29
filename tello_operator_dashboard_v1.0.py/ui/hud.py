import cv2
import numpy as np


def severity_color(level: str):
    if level == "OK":
        return (0, 200, 0)
    if level == "WARN":
        return (0, 200, 255)
    return (0, 0, 255)


def draw_badge(img, label, value, level, x, y, w=260, h=34):
    c = severity_color(level)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
    cv2.putText(img, f"{label}", (x + 10, y + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"{value}", (x + 120, y + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def sparkline(img, values, x, y, w, h, vmin=None, vmax=None, color=(255, 255, 255), label=None):
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        if label:
            cv2.putText(img, label, (x+6, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return

    if vmin is None:
        vmin = min(vals)
    if vmax is None:
        vmax = max(vals)
    if vmax == vmin:
        vmax = vmin + 1e-6

    n = len(values)
    pts = []
    for i, v in enumerate(values):
        if v is None:
            continue
        px = x + int(i * (w - 2) / max(1, n - 1)) + 1
        t = (v - vmin) / (vmax - vmin)
        py = y + h - int(t * (h - 2)) - 1
        pts.append((px, py))

    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2)

    if label:
        cv2.putText(img, label, (x+6, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

def draw_controls_overlay(frame, controls):
    overlay = frame.copy()

    x, y = 30, 40
    line_h = 24

    cv2.rectangle(
        overlay,
        (20, 20),
        (420, 20 + line_h * (len(controls) + 2)),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        overlay,
        "CONTROLS (press H to hide)",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    y += line_h * 1.5

    for key, label, _ in controls:
        cv2.putText(
            overlay,
            f"[{key}]  {label}",
            (x, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        y += line_h

    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
