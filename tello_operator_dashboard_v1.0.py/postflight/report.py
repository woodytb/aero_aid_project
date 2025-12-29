# postflight/report.py
import time
import numpy as np
import cv2


def _safe_array(xs):
    # None -> np.nan
    return np.array([np.nan if x is None else float(x) for x in xs], dtype=np.float32)


def _finite(arr):
    return arr[np.isfinite(arr)]


def _fmt(v, unit="", digits=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if digits == 0:
        return f"{v:.0f}{unit}"
    return f"{v:.{digits}f}{unit}"


def _stat(arr):
    f = _finite(arr)
    if f.size == 0:
        return None, None, None
    return float(np.min(f)), float(np.mean(f)), float(np.max(f))


def _draw_round_rect(img, x1, y1, x2, y2, color, radius=16, thickness=-1):
    # Rounded rectangle using circles + rects
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    r = int(max(2, radius))
    if thickness < 0:
        # fill
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        # outline (simple: draw filled then another smaller cut-out is too much; keep simple)
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
        cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
        cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
        cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)


def _draw_shadow_card(img, x1, y1, x2, y2, bg=(22, 22, 26), shadow=(0, 0, 0)):
    # shadow
    _draw_round_rect(img, x1 + 4, y1 + 6, x2 + 4, y2 + 6, shadow, radius=18, thickness=-1)
    # card
    _draw_round_rect(img, x1, y1, x2, y2, bg, radius=18, thickness=-1)


def _draw_kpi(img, x1, y1, x2, y2, title, value, subtitle="", accent=(80, 170, 255), warn=False):
    _draw_shadow_card(img, x1, y1, x2, y2, bg=(22, 22, 26), shadow=(0, 0, 0))
    # accent line (clean, no thick end-caps)
    bar_h = 4
    cv2.rectangle(img, (x1 + 14, y1 + 12), (x2 - 14, y1 + 12 + bar_h), accent, -1)

    # title
    cv2.putText(img, title, (x1 + 16, y1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 210, 210), 2, cv2.LINE_AA)

    # value
    vcol = (70, 210, 120) if not warn else (70, 70, 255)
    cv2.putText(img, str(value), (x1 + 16, y1 + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, vcol, 3, cv2.LINE_AA)

    if subtitle:
        cv2.putText(img, subtitle, (x1 + 16, y1 + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (170, 170, 170), 2, cv2.LINE_AA)


def _plot_series(img, x, y, x0, y0, w, h, title, unit="", y_min=None, y_max=None,
                 line_color=(255, 255, 255), grid_color=(60, 60, 60), axis_color=(130, 130, 130)):
    # container
    _draw_shadow_card(img, x0, y0, x0 + w, y0 + h, bg=(18, 18, 22), shadow=(0, 0, 0))

    pad_l = 52
    pad_r = 18
    pad_t = 44
    pad_b = 34

    gx1 = x0 + pad_l
    gy1 = y0 + pad_t
    gx2 = x0 + w - pad_r
    gy2 = y0 + h - pad_b

    # title
    cv2.putText(img, title, (x0 + 16, y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA)

    good = np.isfinite(y) & np.isfinite(x)
    if good.sum() < 2:
        cv2.putText(img, "no data", (x0 + 16, y0 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 2, cv2.LINE_AA)
        return

    xx = x[good]
    yy = y[good]

    if y_min is None:
        y_min = float(np.min(yy))
    if y_max is None:
        y_max = float(np.max(yy))
    if abs(y_max - y_min) < 1e-6:
        y_max = y_min + 1.0

    x_min = float(np.min(xx))
    x_max = float(np.max(xx))
    if abs(x_max - x_min) < 1e-6:
        x_max = x_min + 1.0

    # grid
    for i in range(6):
        yline = int(gy1 + (gy2 - gy1) * i / 5)
        cv2.line(img, (gx1, yline), (gx2, yline), grid_color, 1, cv2.LINE_AA)
    for i in range(6):
        xline = int(gx1 + (gx2 - gx1) * i / 5)
        cv2.line(img, (xline, gy1), (xline, gy2), grid_color, 1, cv2.LINE_AA)

    # axis box
    cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (90, 90, 90), 1)

    # labels (min/max)
    cv2.putText(img, f"{_fmt(y_max, unit, 0)}", (x0 + 16, gy1 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 2, cv2.LINE_AA)
    cv2.putText(img, f"{_fmt(y_min, unit, 0)}", (x0 + 16, gy2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 2, cv2.LINE_AA)

    # x ticks: show 0..duration
    dur = x_max - x_min
    cv2.putText(img, "0s", (gx1 - 8, y0 + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 2, cv2.LINE_AA)
    cv2.putText(img, f"{dur:.0f}s", (gx2 - 42, y0 + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 2, cv2.LINE_AA)

    # map to pixels
    pts = []
    for xi, yi in zip(xx, yy):
        px = int(gx1 + (xi - x_min) / (x_max - x_min) * (gx2 - gx1))
        py = int(gy2 - (yi - y_min) / (y_max - y_min) * (gy2 - gy1))
        pts.append((px, py))

    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, line_color, 2, cv2.LINE_AA)

    # last value badge
    last_val = float(yy[-1])
    badge = f"{_fmt(last_val, unit, 1)}"
    tw = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
    bx2 = gx2 - 10
    bx1 = bx2 - tw - 18
    by2 = gy1 + 26
    by1 = by2 - 22
    _draw_round_rect(img, bx1, by1, bx2, by2, (30, 30, 34), radius=10, thickness=-1)
    cv2.putText(img, badge, (bx1 + 9, by2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)


def build_postflight_image(samples, width=1280, height=720):
    """
    samples: list of dicts with keys:
      t, bat, h, tof, vxy, yaw, pitch, roll, state_age, video_age
    Returns: BGR image (np.uint8)
    """
    if not samples:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img, "No post-flight data collected.", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    # time base
    t0 = samples[0]["t"]
    ts = np.array([s["t"] - t0 for s in samples], dtype=np.float32)
    dur = float(ts[-1]) if len(ts) else 0.0

    # series
    bat = _safe_array([s.get("bat") for s in samples])
    h = _safe_array([s.get("h") for s in samples])
    tof = _safe_array([s.get("tof") for s in samples])
    vxy = _safe_array([s.get("vxy") for s in samples])
    state_age = _safe_array([s.get("state_age") for s in samples])
    video_age = _safe_array([s.get("video_age") for s in samples])

    # stats
    bat_min, bat_avg, bat_max = _stat(bat)
    h_min, h_avg, h_max = _stat(h)
    tof_min, tof_avg, tof_max = _stat(tof)
    vxy_min, vxy_avg, vxy_max = _stat(vxy)

    sa_min, sa_avg, sa_max = _stat(state_age)
    va_min, va_avg, va_max = _stat(video_age)

    bat_drop = None
    fbat = _finite(bat)
    if fbat.size >= 2:
        bat_drop = float(fbat[0] - fbat[-1])

    # background (subtle gradient)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        v = int(18 + 18 * (y / max(1, height - 1)))
        img[y, :, :] = (v, v, v)

    # header
    cv2.putText(img, "POST-FLIGHT DASHBOARD", (40, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.35, (245, 245, 245), 3, cv2.LINE_AA)
    cv2.putText(
        img,
        f"Duration: {dur:.1f}s   Samples: {len(samples)}   Generated: {time.strftime('%H:%M:%S')}",
        (40, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 190, 190), 2, cv2.LINE_AA
    )

    # KPI cards row
    kpi_y1, kpi_y2 = 110, 230
    kpi_w = 300
    gap = 18
    x = 40

    # warning flags
    warn_state = (sa_max is not None and sa_max > 2.0)
    warn_video = (va_max is not None and va_max > 2.0)

    _draw_kpi(
        img, x, kpi_y1, x + kpi_w, kpi_y2,
        "Battery", _fmt(bat_avg, "%", 0),
        subtitle=f"min {_fmt(bat_min, '%', 0)}  max {_fmt(bat_max, '%', 0)}  drop {_fmt(bat_drop, '%', 0)}",
        accent=(80, 170, 255),
        warn=False
    )
    x += kpi_w + gap

    _draw_kpi(
        img, x, kpi_y1, x + kpi_w, kpi_y2,
        "Max Height", _fmt(h_max, "cm", 0),
        subtitle=f"avg {_fmt(h_avg, 'cm', 0)}  min {_fmt(h_min, 'cm', 0)}",
        accent=(255, 170, 80),
        warn=False
    )
    x += kpi_w + gap

    _draw_kpi(
        img, x, kpi_y1, x + kpi_w, kpi_y2,
        "Max Speed", _fmt(vxy_max, "cm/s", 0),
        subtitle=f"avg {_fmt(vxy_avg, 'cm/s', 0)}  min {_fmt(vxy_min, 'cm/s', 0)}",
        accent=(120, 220, 140),
        warn=False
    )
    x += kpi_w + gap

    # connectivity card
    conn_value = f"STATE {_fmt(sa_max, 's', 1)} | VIDEO {_fmt(va_max, 's', 1)}"
    conn_sub = f"avg: state {_fmt(sa_avg, 's', 1)}  video {_fmt(va_avg, 's', 1)}"
    _draw_kpi(
        img, x, kpi_y1, x + kpi_w, kpi_y2,
        "Link Age (max)", conn_value,
        subtitle=conn_sub,
        accent=(200, 120, 255),
        warn=(warn_state or warn_video)
    )

    # charts area (2x2)
    chart_w = 600
    chart_h = 220
    left_x = 40
    right_x = 680
    top_y = 250
    bottom_y = 490

    _plot_series(img, ts, bat, left_x, top_y, chart_w, chart_h, "Battery", unit="%", y_min=0, y_max=100,
                 line_color=(80, 170, 255))
    _plot_series(img, ts, h, right_x, top_y, chart_w, chart_h, "Height", unit="cm",
                 line_color=(255, 170, 80))
    _plot_series(img, ts, tof, left_x, bottom_y, chart_w, chart_h, "TOF", unit="cm",
                 line_color=(200, 200, 200))
    _plot_series(img, ts, vxy, right_x, bottom_y, chart_w, chart_h, "Vxy Speed", unit="cm/s",
                 line_color=(120, 220, 140))

    # footer
    cv2.putText(img, "ESC / Q to close   |   Tip: Press Q to close this window after review.",
                (40, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 190, 190), 2, cv2.LINE_AA)

    return img


def show_postflight_dashboard(samples):
    img = build_postflight_image(samples)
    win = "Post-Flight Dashboard"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    # optional: save a PNG automatically for documentation
    try:
        fn = f"postflight_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(fn, img)
        # not spamming console too much is fine, but this is useful:
        print(f"[POSTFLIGHT] saved {fn}")
    except Exception:
        pass

    while True:
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q"), ord("Q")):  # ESC or Q
            break
    cv2.destroyWindow(win)
