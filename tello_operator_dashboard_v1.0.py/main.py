import time
import cv2
import numpy as np
from collections import deque
from servo.servo_udp import ServoUDP

from tello.client import TelloClient, KeepAlive
from video.capture import VideoManager
from utils.logger import CsvLogger
from utils.telemetry import (
    to_int, classify_age, classify_bat, tof_level,
    STATE_WARN_AGE_S, STATE_CRIT_AGE_S, VIDEO_WARN_AGE_S, VIDEO_CRIT_AGE_S
)
from servo.servo_serial import ServoSerial
from missions.definitions import DEFAULT_MISSIONS
from ui.hud import draw_badge, sparkline
from vision.aruco_nav import MissionRunner, ARUCO_AVAILABLE
from postflight.report import show_postflight_dashboard


def main():
    t = TelloClient()
    keepalive = KeepAlive(t)
    keepalive.enabled = True

    logger = CsvLogger()
    # Nur SERIAL!
    # servo = ServoSerial(baud=115200)
    # servo.auto_connect()

    servo = ServoUDP(host="192.168.10.X", port=8890)  # <-- ESP32-IP hier rein
    servo.auto_connect()

    # UI first
    win = "Tello Operator PRO (structured)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    video = VideoManager(video_port=11111)

    # ---------------------------
    # Controls overlay toggle + list
    # ---------------------------
    show_controls = False

    # ---------------------------
    # Servo angle controls (conflict-free with 0-9 target keys)
    # Arduino expects: OPEN / CLOSE / ANGLE <number>
    # ---------------------------
    servo_angle = 90
    SERVO_STEP = 10
    SERVO_MIN = 0
    SERVO_MAX = 180

    # ---------------------------
    # NEW: Flight + manual RC config
    # ---------------------------
    flying = False

    RC_STEP = 35          # normal speed
    RC_STEP_FAST = 60     # fast speed (uppercase keys)
    RC_BURST_S = 0.12     # burst duration per keypress
    RC_MIN_INTERVAL_S = 0.06
    last_rc_sent = 0.0

    CONTROLS_LIST = [
        ("Q", "Quit"),
        ("Z", "Arm / Disarm toggle"),
        ("T", "Takeoff (needs ARMED + SDK OK)"),
        ("L", "Land (needs ARMED + SDK OK)"),
        ("W/A/S/D", "Manual move (forward/left/back/right)"),
        ("E/F", "Manual up/down"),
        ("J/K", "Yaw left/right"),
        ("(Uppercase)", "Faster manual speed (WASD/E/F/J/K)"),
        ("C", "SDK: send 'command' (re-init)"),
        ("V", "Video retry (streamon + reopen)"),
        ("R", "Record CSV toggle"),
        ("I", "Snapshot (save jpg)"),
        ("0-9", "Set target ArUco ID"),
        ("G", "Goto target ID (Aruco nav)"),
        ("N", "Next mission"),
        ("M", "Run selected mission"),
        ("X", "Abort mission"),
        ("H", "Show/Hide this controls list"),
        ("O", "Servo gripper OPEN"),
        ("P", "Servo gripper CLOSE"),
        ("[", "Servo ANGLE -10°"),
        ("]", "Servo ANGLE +10°"),
    ]

    def draw_controls_overlay(frame, controls, title="CONTROLS (press H to hide)"):
        """
        Draws a readable, non-cut-off list of all controls.
        Keeps it simple: one column, auto-fits height (up to window),
        and anchors it top-right so your left panel stays visible.
        """
        h_img, w_img = frame.shape[:2]
        pad = 16
        line_h = 22
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Compute overlay size
        max_lines = len(controls) + 2
        box_h = pad * 2 + int(max_lines * line_h)
        box_h = min(box_h, h_img - 2 * pad)

        box_w = 520  # enough for text
        x2 = w_img - pad
        x1 = max(pad, x2 - box_w)
        y1 = pad
        y2 = y1 + box_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)

        # Title
        tx = x1 + 12
        ty = y1 + 28
        cv2.putText(frame, title, (tx, ty), font, 0.7, (0, 255, 255), 2)

        # List items
        ty += 30
        for key, desc in controls:
            if ty > y2 - 12:
                # stop drawing if we run out of space
                cv2.putText(frame, "...", (tx, y2 - 12), font, 0.7, (255, 255, 255), 2)
                break
            cv2.putText(frame, f"[{key}]  {desc}", (tx, ty), font, 0.6, (255, 255, 255), 2)
            ty += line_h

        return frame

    hist_len = 150
    bat_hist = deque([None] * hist_len, maxlen=hist_len)
    h_hist = deque([None] * hist_len, maxlen=hist_len)
    tof_hist = deque([None] * hist_len, maxlen=hist_len)
    vxy_hist = deque([None] * hist_len, maxlen=hist_len)

    armed = False
    sdk_ok = False

    # shared frame for aruco runner
    latest_frame = {"img": None}
    selected_aruco_id = 0
    missions = list(DEFAULT_MISSIONS)
    mission_idx = 0
    session_samples = []  # <-- vor der while True loop anlegen

    def set_status(msg: str):
        # store in tello client reply area
        with t.reply_lock:
            t.last_cmd_reply = msg

    def ensure_sdk():
        nonlocal sdk_ok
        if sdk_ok:
            return True
        sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)
        return sdk_ok

    def get_frame_copy():
        img = latest_frame["img"]
        if img is None:
            return None
        return img.copy()

    def get_stream_ok():
        return video.stream_ok

    def is_armed():
        return armed

    # ---------------------------
    # NEW: manual rc burst sender
    # rc lr fb ud yaw  (each -100..100)
    # ---------------------------
    def send_rc(lr=0, fb=0, ud=0, yaw=0, duration_s=RC_BURST_S):
        nonlocal last_rc_sent
        if not armed:
            set_status("rc blocked: DISARMED (press Z)")
            return False
        if not ensure_sdk():
            set_status("rc blocked: SDK NOT READY (press C)")
            return False
        if not flying:
            set_status("rc blocked: not flying (press T)")
            return False

        now_local = time.time()
        if now_local - last_rc_sent < RC_MIN_INTERVAL_S:
            return False
        last_rc_sent = now_local

        lr = int(max(-100, min(100, lr)))
        fb = int(max(-100, min(100, fb)))
        ud = int(max(-100, min(100, ud)))
        yaw = int(max(-100, min(100, yaw)))

        end_t = now_local + float(duration_s)
        while time.time() < end_t:
            t.send(f"rc {lr} {fb} {ud} {yaw}", delay=0.0)
            time.sleep(0.04)  # ~25 Hz
        t.send("rc 0 0 0 0", delay=0.0)
        return True

    runner = MissionRunner(
        tello=t,
        get_frame_fn=get_frame_copy,
        get_stream_ok_fn=get_stream_ok,
        is_armed_fn=is_armed,
        ensure_sdk_fn=ensure_sdk,
        set_status_fn=set_status
    )

    # init SDK + stream (non-fatal)
    t.send("command", delay=0.3)
    sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)
    t.send("streamoff", delay=0.2)  # reset video stream
    t.send("streamon", delay=0.3)

    time.sleep(0.5)
    video.open()  # open after streamon to avoid ffmpeg timeout

    if not video.cap or not video.cap.isOpened():
        for _ in range(3):
            time.sleep(0.5)
            if video.retry_if_needed(every_s=0.0):
                break

    last_key_action = 0.0
    KEY_COOLDOWN_S = 0.25

    # ---------------------------
    # NEW: for diagnosing unexpected exits
    # ---------------------------
    loop_error_count = 0

    try:
        while True:
            try:
                now = time.time()

                # If user closes the window, exit cleanly (prevents weird states)
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    # on some OpenCV builds this can throw if window not ready
                    pass

                ret, frame = video.read()
                if ret and frame is not None:
                    latest_frame["img"] = frame
                else:
                    if not video.stream_ok:
                        video.retry_if_needed(every_s=3.0)
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(frame, "Dashboard läuft - kein Video-Stream.", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                st = t.get_state_copy()

                bat = to_int(st, "bat", None)
                h = to_int(st, "h", None)
                tof = to_int(st, "tof", None)
                vgx = to_int(st, "vgx", None)
                vgy = to_int(st, "vgy", None)

                # ---------------------------
                # NEW: robust flying detection from h/tof
                # This prevents "blocked: not flying" even if takeoff reply is lost.
                # ---------------------------
                if h is not None:
                    if h >= 15:
                        flying = True
                    elif h <= 5:
                        flying = False
                elif tof is not None:
                    if tof >= 20:
                        flying = True
                    elif tof <= 10:
                        flying = False

                bat_hist.append(bat)
                h_hist.append(h)
                tof_hist.append(tof)
                if vgx is not None and vgy is not None:
                    vxy_hist.append((vgx**2 + vgy**2) ** 0.5)
                else:
                    vxy_hist.append(None)

                logger.write_state(st)

                state_age = now - t.last_state_ts if t.last_state_ts > 0 else 999.0
                video_age = now - video.last_frame_ts if video.last_frame_ts > 0 else 999.0
                vxy_now = vxy_hist[-1] if len(vxy_hist) else None
                session_samples.append({
                    "t": now,
                    "bat": bat,
                    "h": h,
                    "tof": tof,
                    "vxy": vxy_now,
                    "yaw": to_int(st, "yaw", None),
                    "pitch": to_int(st, "pitch", None),
                    "roll": to_int(st, "roll", None),
                    "state_age": state_age,
                    "video_age": video_age,
                })

                state_level = classify_age(state_age, STATE_WARN_AGE_S, STATE_CRIT_AGE_S)
                video_level = classify_age(video_age, VIDEO_WARN_AGE_S, VIDEO_CRIT_AGE_S)
                bat_level = classify_bat(bat)
                tof_lv = tof_level(tof)

                h_img, w_img = frame.shape[:2]
                panel_w = 560
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (panel_w, h_img), (0, 0, 0), -1)
                frame[:] = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

                draw_badge(frame, "MODE", "ARMED" if armed else "DISARMED", "OK" if armed else "WARN", 18, 55)
                draw_badge(frame, "FLIGHT", "FLYING" if flying else "LANDED", "OK", 288, 55)
                draw_badge(frame, "SDK", "OK" if sdk_ok else "NOT READY", "OK" if sdk_ok else "WARN", 18, 95)
                draw_badge(frame, "STATE", f"{state_level} ({state_age:.1f}s)", state_level, 18, 135)
                draw_badge(frame, "VIDEO", f"{video_level} ({video_age:.1f}s)", video_level, 18, 175)
                draw_badge(frame, "BAT", f"{bat if bat is not None else '?'}%", bat_level, 18, 215)
                draw_badge(frame, "H", f"{h if h is not None else '?'} cm", "OK" if h is not None else "WARN", 18, 255)
                draw_badge(frame, "TOF", f"{tof if tof is not None else '?'} cm", tof_lv, 18, 295)

                # show angle in servo badge
                draw_badge(frame, "SERVO",
                           ("OK" if servo.is_connected() else "NOT CONNECTED") + f" ({servo_angle}°)",
                           "OK" if servo.is_connected() else "WARN", 18, 335)

                draw_badge(frame, "ARUCO", "OK" if ARUCO_AVAILABLE else "NO MODULE",
                           "OK" if ARUCO_AVAILABLE else "WARN", 288, 335)

                mname, mids = missions[mission_idx]
                draw_badge(frame, "MISSION", f"{mission_idx}: {mids}", "OK", 18, 375)
                draw_badge(frame, "TARGET", f"ID {selected_aruco_id}", "OK", 288, 375)

                sent, rep = t.get_cmd_status()
                cv2.putText(frame, f"Sent : {sent}", (18, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Reply: {rep}", (18, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                sparkline(frame, list(bat_hist), 18, 510, 250, 70, vmin=0, vmax=100, label="BAT history")
                sparkline(frame, list(h_hist), 288, 510, 250, 70, vmin=0, vmax=200, label="H history")
                sparkline(frame, list(tof_hist), 18, 595, 250, 70, vmin=0, vmax=200, label="TOF history")
                sparkline(frame, list(vxy_hist), 288, 595, 250, 70, vmin=0, vmax=60, label="Vxy cm/s")

                # controls overlay
                if show_controls:
                    frame = draw_controls_overlay(frame, CONTROLS_LIST)

                # Footer: NO key list (everything is in Help). Only status.
                bat_s = f"{bat}%" if bat is not None else "?"
                h_s = f"{h}cm" if h is not None else "?"
                tof_s = f"{tof}cm" if tof is not None else "?"
                vxy_s = f"{vxy_now:.1f}" if isinstance(vxy_now, (int, float)) else "?"
                cv2.putText(
                    frame,
                    f"Status | BAT {bat_s} | H {h_s} | TOF {tof_s} | Vxy {vxy_s} cm/s | state_age {state_age:.1f}s | video_age {video_age:.1f}s",
                    (18, h_img - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
                )

                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 255:
                    continue
                if now - last_key_action < KEY_COOLDOWN_S:
                    continue
                last_key_action = now

                if key in (ord("q"), ord("Q")):
                    break

                elif key in (ord("h"), ord("H")):
                    show_controls = not show_controls
                    set_status("controls: ON" if show_controls else "controls: OFF")

                # ---------------------------
                # Servo controls
                # ---------------------------
                elif key in (ord("o"), ord("O")):
                    ok = servo.send("OPEN")
                    set_status("servo: OPEN" if ok else "servo: NOT CONNECTED")

                elif key in (ord("p"), ord("P")):
                    ok = servo.send("CLOSE")
                    set_status("servo: CLOSE" if ok else "servo: NOT CONNECTED")

                elif key == ord("["):
                    servo_angle = max(SERVO_MIN, servo_angle - SERVO_STEP)
                    ok = servo.send(f"ANGLE {servo_angle}")
                    set_status(f"servo: ANGLE {servo_angle}" if ok else "servo: NOT CONNECTED")

                elif key == ord("]"):
                    servo_angle = min(SERVO_MAX, servo_angle + SERVO_STEP)
                    ok = servo.send(f"ANGLE {servo_angle}")
                    set_status(f"servo: ANGLE {servo_angle}" if ok else "servo: NOT CONNECTED")

                # ---------------------------
                # ARM moved to Z (so WASD is free)
                # ---------------------------
                elif key in (ord("z"), ord("Z")):
                    armed = not armed
                    set_status("ARMED" if armed else "DISARMED")

                # ---------------------------
                # SDK / VIDEO
                # ---------------------------
                elif key in (ord("c"), ord("C")):
                    sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)

                elif key in (ord("v"), ord("V")):
                    t.send("streamon", delay=0.2)
                    video.retry_if_needed(every_s=0.0)
                    set_status(f"video: retry ({video.uri})")

                # ---------------------------
                # TAKEOFF / LAND
                # ---------------------------
                elif key in (ord("t"), ord("T")):
                    if not armed:
                        set_status("takeoff blocked: DISARMED (press Z)")
                    elif not ensure_sdk():
                        set_status("takeoff blocked: SDK NOT READY (press C)")
                    elif flying:
                        set_status("takeoff blocked: already flying")
                    else:
                        ok = t.send_and_wait_ok("takeoff", timeout_s=15.0)
                        if ok:
                            flying = True
                        set_status("TAKEOFF" if ok else "takeoff failed (reply timeout?)")

                elif key in (ord("l"), ord("L")):
                    if not armed:
                        set_status("land blocked: DISARMED (press Z)")
                    elif not ensure_sdk():
                        set_status("land blocked: SDK NOT READY (press C)")
                    elif not flying:
                        set_status("land blocked: not flying")
                    else:
                        ok = t.send_and_wait_ok("land", timeout_s=15.0)
                        if ok:
                            flying = False
                        set_status("LAND" if ok else "land failed (reply timeout?)")

                # ---------------------------
                # RECORD
                # ---------------------------
                elif key in (ord("r"), ord("R")):
                    if not logger.enabled:
                        logger.start()
                        set_status(f"recording: ON ({logger.filename})")
                    else:
                        fn = logger.filename
                        logger.stop()
                        set_status(f"recording: OFF ({fn})")

                # ---------------------------
                # SNAPSHOT moved to I (so S is free)
                # ---------------------------
                elif key in (ord("i"), ord("I")):
                    fn = f"tello_snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(fn, frame)
                    set_status(f"snapshot saved: {fn}")

                # ---------------------------
                # Manual flight: WASD / E/F / J/K
                # Uppercase => faster
                # ---------------------------
                elif key in (ord("w"), ord("W")):
                    step = RC_STEP_FAST if key == ord("W") else RC_STEP
                    send_rc(fb=+step)

                elif key in (ord("s"), ord("S")):
                    step = RC_STEP_FAST if key == ord("S") else RC_STEP
                    send_rc(fb=-step)

                elif key in (ord("a"), ord("A")):
                    step = RC_STEP_FAST if key == ord("A") else RC_STEP
                    send_rc(lr=-step)

                elif key in (ord("d"), ord("D")):
                    step = RC_STEP_FAST if key == ord("D") else RC_STEP
                    send_rc(lr=+step)

                elif key in (ord("e"), ord("E")):
                    step = RC_STEP_FAST if key == ord("E") else RC_STEP
                    send_rc(ud=+step)

                elif key in (ord("f"), ord("F")):
                    step = RC_STEP_FAST if key == ord("F") else RC_STEP
                    send_rc(ud=-step)

                elif key in (ord("j"), ord("J")):
                    step = RC_STEP_FAST if key == ord("J") else RC_STEP
                    send_rc(yaw=-step)

                elif key in (ord("k"), ord("K")):
                    step = RC_STEP_FAST if key == ord("K") else RC_STEP
                    send_rc(yaw=+step)

                # ---------------------------
                # MISSIONS / ARUCO
                # ---------------------------
                elif key in (ord("x"), ord("X")):
                    runner.abort()

                elif ord("0") <= key <= ord("9"):
                    selected_aruco_id = int(chr(key))
                    set_status(f"target set: ArUco ID {selected_aruco_id}")

                elif key in (ord("g"), ord("G")):
                    runner.goto_id(selected_aruco_id)

                elif key in (ord("n"), ord("N")):
                    mission_idx = (mission_idx + 1) % len(missions)
                    set_status(f"mission selected: {missions[mission_idx][0]}")

                elif key in (ord("m"), ord("M")):
                    _, ids_list = missions[mission_idx]
                    runner.run_mission(ids_list)

            except Exception as e:
                # NEW: never silently exit; log and keep running
                loop_error_count += 1
                msg = f"LOOP ERROR #{loop_error_count}: {type(e).__name__}: {e}"
                set_status(msg)
                print(msg)
                # small delay to avoid spamming if error repeats fast
                time.sleep(0.2)
                continue

    finally:
        try:
            t.send("streamoff", delay=0.2)
        except Exception:
            pass
        try:
            video.release()
        except Exception:
            pass
        try:
            logger.stop()
        except Exception:
            pass
        try:
            servo.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        show_postflight_dashboard(session_samples)
        t.close()


if __name__ == "__main__":
    main()
