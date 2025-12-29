import socket
import threading
import time
import cv2
import numpy as np
from collections import deque
import csv
import os

# NEW:
import serial
from serial.tools import list_ports

TELLO_IP = "192.168.10.1"
CMD_PORT = 8889
STATE_PORT = 8890
VIDEO_PORT = 11111
LOCAL_CMD_PORT = 9000

KEY_COOLDOWN_S = 0.25
LAND_DOUBLE_PRESS_WINDOW_S = 1.0

BAT_WARN = 30
BAT_CRIT = 20

STATE_WARN_AGE_S = 1.0
STATE_CRIT_AGE_S = 2.5

VIDEO_WARN_AGE_S = 1.0
VIDEO_CRIT_AGE_S = 2.5

TOF_MIN_VALID = 5
TOF_MAX_VALID = 300

running = True

latest_state = {}
state_lock = threading.Lock()
last_state_ts = 0.0

last_cmd_reply = "—"
last_sent_cmd = "—"
reply_lock = threading.Lock()


def ts():
    return time.strftime("%H:%M:%S")


def parse_state(state_str: str) -> dict:
    out = {}
    for part in state_str.strip().split(";"):
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k] = v
    return out


def to_int(d, key, default=None):
    try:
        return int(float(d.get(key, default)))
    except Exception:
        return default


def severity_color(level: str):
    if level == "OK":
        return (0, 200, 0)
    if level == "WARN":
        return (0, 200, 255)
    return (0, 0, 255)


def classify_age(age_s, warn_s, crit_s):
    if age_s >= crit_s:
        return "CRIT"
    if age_s >= warn_s:
        return "WARN"
    return "OK"


def classify_bat(bat):
    if bat is None:
        return "WARN"
    if bat <= BAT_CRIT:
        return "CRIT"
    if bat <= BAT_WARN:
        return "WARN"
    return "OK"


def open_video_capture():
    candidates = [
        ("udp://@:11111", cv2.CAP_FFMPEG),
        (f"udp://0.0.0.0:{VIDEO_PORT}", cv2.CAP_FFMPEG),
        (f"udp://127.0.0.1:{VIDEO_PORT}", cv2.CAP_FFMPEG),
    ]
    for uri, backend in candidates:
        cap = cv2.VideoCapture(uri, backend)
        time.sleep(0.25)
        if cap.isOpened():
            return cap, uri
        cap.release()
    return cv2.VideoCapture(), "none"


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


class CsvLogger:
    def __init__(self):
        self.enabled = False
        self.fp = None
        self.writer = None
        self.filename = None

    def start(self):
        os.makedirs("logs", exist_ok=True)
        self.filename = os.path.join("logs", f"tello_log_{int(time.time())}.csv")
        self.fp = open(self.filename, "w", newline="")
        self.writer = csv.writer(self.fp)
        self.writer.writerow([
            "ts", "bat", "h", "tof", "yaw", "pitch", "roll",
            "vgx", "vgy", "vgz", "templ", "temph", "baro", "time"
        ])
        self.enabled = True

    def stop(self):
        self.enabled = False
        try:
            if self.fp:
                self.fp.close()
        except Exception:
            pass
        self.fp = None
        self.writer = None

    def write_state(self, st):
        if not self.enabled or not self.writer:
            return
        self.writer.writerow([
            time.time(),
            st.get("bat"), st.get("h"), st.get("tof"),
            st.get("yaw"), st.get("pitch"), st.get("roll"),
            st.get("vgx"), st.get("vgy"), st.get("vgz"),
            st.get("templ"), st.get("temph"), st.get("baro"), st.get("time")
        ])


class TelloClient:
    def __init__(self):
        global last_state_ts
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind(("", LOCAL_CMD_PORT))
        self.cmd_sock.settimeout(2.0)
        self.tello_addr = (TELLO_IP, CMD_PORT)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind(("", STATE_PORT))
        self.state_sock.settimeout(1.0)

        last_state_ts = 0.0

        self.cmd_thread = threading.Thread(target=self._cmd_recv_loop, daemon=True)
        self.cmd_thread.start()

        self.state_thread = threading.Thread(target=self._state_recv_loop, daemon=True)
        self.state_thread.start()

    def _cmd_recv_loop(self):
        global last_cmd_reply
        while running:
            try:
                data, _ = self.cmd_sock.recvfrom(1024)
                msg = data.decode("utf-8", errors="ignore").strip()
                print(f"[{ts()}] TELLO REPLY: {msg}")
                with reply_lock:
                    last_cmd_reply = msg
            except socket.timeout:
                continue
            except OSError:
                break

    def _state_recv_loop(self):
        global latest_state, last_state_ts
        while running:
            try:
                data, _ = self.state_sock.recvfrom(2048)
                st = parse_state(data.decode("utf-8", errors="ignore"))
                with state_lock:
                    latest_state = st
                last_state_ts = time.time()
            except socket.timeout:
                continue
            except OSError:
                break

    def send(self, cmd: str, delay: float = 0.12):
        global last_sent_cmd, last_cmd_reply
        try:
            self.cmd_sock.sendto(cmd.encode("utf-8"), self.tello_addr)
            print(f"[{ts()}] TELLO CMD: {cmd}")
            with reply_lock:
                last_sent_cmd = cmd
        except OSError as e:
            print(f"[{ts()}] SEND ERROR: {e}")
            with reply_lock:
                last_cmd_reply = f"SEND ERROR: {e}"
        time.sleep(delay)

    def send_and_wait_ok(self, cmd: str, timeout_s: float = 2.0) -> bool:
        global last_cmd_reply
        with reply_lock:
            last_cmd_reply = "—"
        self.send(cmd, delay=0.05)

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with reply_lock:
                r = str(last_cmd_reply).lower()
            if r == "ok":
                return True
            if r.startswith("error"):
                return False
            time.sleep(0.05)
        return False

    def close(self):
        try:
            self.cmd_sock.close()
        except Exception:
            pass
        try:
            self.state_sock.close()
        except Exception:
            pass


class KeepAlive:
    def __init__(self, client: TelloClient):
        self.client = client
        self.enabled = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while running:
            if self.enabled:
                self.client.send("command", delay=0.05)
            time.sleep(5.0)


# ---------- Serial / Servo ----------
class ServoSerial:
    def __init__(self, baud=115200):
        self.ser = None
        self.port = None
        self.baud = baud

    def auto_connect(self):
        """
        Tries to find an ESP32 / USB serial port on macOS.
        """
        ports = list(list_ports.comports())
        candidates = []
        for p in ports:
            dev = p.device
            # macOS typical patterns
            if dev.startswith("/dev/tty.usbmodem") or dev.startswith("/dev/tty.usbserial"):
                candidates.append(dev)

        # If only one, pick it
        for dev in candidates:
            try:
                s = serial.Serial(dev, self.baud, timeout=0.2)
                self.ser = s
                self.port = dev
                return True
            except Exception:
                continue

        return False

    def is_connected(self):
        return self.ser is not None and self.ser.is_open

    def send(self, line: str):
        if not self.is_connected():
            return False
        try:
            msg = (line.strip() + "\n").encode("utf-8")
            self.ser.write(msg)
            return True
        except Exception:
            return False

    def close(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None
        self.port = None


def main():
    global running, last_cmd_reply, last_sent_cmd, last_state_ts

    t = TelloClient()
    logger = CsvLogger()

    servo = ServoSerial(baud=115200)
    servo_connected = servo.auto_connect()
    servo_status = f"SERVO: {'OK' if servo_connected else 'NOT CONNECTED'}"

    hist_len = 150
    bat_hist = deque([None] * hist_len, maxlen=hist_len)
    h_hist = deque([None] * hist_len, maxlen=hist_len)
    tof_hist = deque([None] * hist_len, maxlen=hist_len)
    vxy_hist = deque([None] * hist_len, maxlen=hist_len)

    armed = False
    sdk_ok = False

    last_key_action = 0.0
    last_land_key_ts = 0.0

    # Init SDK + video
    t.send("command", delay=0.5)
    sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)
    t.send("streamon", delay=0.5)
    t.send("streamon", delay=0.5)

    cap, uri = open_video_capture()
    print(f"[{ts()}] Video capture URI: {uri} | opened: {cap.isOpened()}")

    keepalive = KeepAlive(t)
    keepalive.enabled = True

    win = "Tello Operator PRO + SERVO"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    stream_ok = False
    last_frame_ts = 0.0
    fps_hist = deque([], maxlen=60)
    prev_frame_ts = None

    def mission_pickup_drop():
        """
        land -> close -> takeoff -> right 100 -> land -> open
        """
        nonlocal servo_status, servo_connected, sdk_ok

        if not armed:
            with reply_lock:
                last_cmd_reply = "mission blocked: DISARMED"
            return
        if not sdk_ok:
            sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)
            if not sdk_ok:
                with reply_lock:
                    last_cmd_reply = "mission blocked: SDK not ready"
                return
        if not servo.is_connected():
            servo_connected = servo.auto_connect()
            servo_status = f"SERVO: {'OK' if servo_connected else 'NOT CONNECTED'}"
            if not servo.is_connected():
                with reply_lock:
                    last_cmd_reply = "mission blocked: servo not connected"
                return

        # land
        ok = t.send_and_wait_ok("land", timeout_s=8.0)
        if not ok:
            with reply_lock:
                last_cmd_reply = "mission: land failed"
            return

        time.sleep(1.0)

        # close gripper
        if servo.send("CLOSE"):
            with reply_lock:
                last_cmd_reply = "mission: CLOSE sent"
        else:
            with reply_lock:
                last_cmd_reply = "mission: CLOSE failed"
            return

        time.sleep(1.0)

        # takeoff
        ok = t.send_and_wait_ok("takeoff", timeout_s=8.0)
        if not ok:
            with reply_lock:
                last_cmd_reply = "mission: takeoff failed"
            return

        # move right 1m
        ok = t.send_and_wait_ok("right 100", timeout_s=5.0)
        if not ok:
            with reply_lock:
                last_cmd_reply = "mission: right 100 failed"
            return

        # land
        ok = t.send_and_wait_ok("land", timeout_s=8.0)
        if not ok:
            with reply_lock:
                last_cmd_reply = "mission: land2 failed"
            return

        time.sleep(1.0)

        # open gripper
        if servo.send("OPEN"):
            with reply_lock:
                last_cmd_reply = "mission: OPEN sent"
        else:
            with reply_lock:
                last_cmd_reply = "mission: OPEN failed"

    try:
        while True:
            now = time.time()

            # Video
            ret, frame = (False, None)
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()

            if ret and frame is not None:
                stream_ok = True
                last_frame_ts = now
                if prev_frame_ts is not None:
                    dt = now - prev_frame_ts
                    if dt > 0:
                        fps_hist.append(1.0 / dt)
                prev_frame_ts = now
            else:
                if now - last_frame_ts > 2.0:
                    stream_ok = False
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for video... (WLAN? streamon? Firewall?)",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # State
            with state_lock:
                st = dict(latest_state)

            bat = to_int(st, "bat", None)
            h = to_int(st, "h", None)
            tof = to_int(st, "tof", None)
            vgx = to_int(st, "vgx", None)
            vgy = to_int(st, "vgy", None)

            bat_hist.append(bat)
            h_hist.append(h)
            tof_hist.append(tof)
            if vgx is not None and vgy is not None:
                vxy_hist.append((vgx**2 + vgy**2) ** 0.5)
            else:
                vxy_hist.append(None)

            logger.write_state(st)

            # Health
            state_age = now - last_state_ts if last_state_ts > 0 else 999.0
            video_age = now - last_frame_ts if last_frame_ts > 0 else 999.0
            state_level = classify_age(state_age, STATE_WARN_AGE_S, STATE_CRIT_AGE_S)
            video_level = classify_age(video_age, VIDEO_WARN_AGE_S, VIDEO_CRIT_AGE_S)
            bat_level = classify_bat(bat)
            tof_level = "OK"
            if tof is None:
                tof_level = "WARN"
            elif tof < TOF_MIN_VALID or tof > TOF_MAX_VALID:
                tof_level = "WARN"

            fps = int(sum(fps_hist) / len(fps_hist)) if len(fps_hist) else 0

            # Panel
            h_img, w_img = frame.shape[:2]
            panel_w = 560
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (panel_w, h_img), (0, 0, 0), -1)
            frame[:] = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

            cv2.putText(frame, "TELLO OPERATOR PRO + SERVO", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

            draw_badge(frame, "MODE", "ARMED" if armed else "DISARMED", "OK" if armed else "WARN", 18, 55)
            draw_badge(frame, "SDK", "OK" if sdk_ok else "NOT READY", "OK" if sdk_ok else "WARN", 18, 95)

            draw_badge(frame, "STATE", f"{state_level} ({state_age:.1f}s)", state_level, 18, 135)
            draw_badge(frame, "VIDEO", f"{video_level} ({video_age:.1f}s)", video_level, 18, 175)

            draw_badge(frame, "BAT", f"{bat if bat is not None else '?'}%", bat_level, 18, 215)
            draw_badge(frame, "H", f"{h if h is not None else '?'} cm", "OK" if h is not None else "WARN", 18, 255)
            draw_badge(frame, "TOF", f"{tof if tof is not None else '?'} cm", tof_level, 18, 295)

            # Servo status badge
            servo_level = "OK" if servo.is_connected() else "WARN"
            draw_badge(frame, "SERVO", "OK" if servo.is_connected() else "NOT CONNECTED", servo_level, 18, 335)

            with reply_lock:
                sent = str(last_sent_cmd)
                rep = str(last_cmd_reply)

            cv2.putText(frame, f"Sent : {sent}", (18, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Reply: {rep}", (18, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"ATT  yaw {st.get('yaw','?')}  pitch {st.get('pitch','?')}  roll {st.get('roll','?')}",
                        (18, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"VEL  vgx {st.get('vgx','?')}  vgy {st.get('vgy','?')}  vgz {st.get('vgz','?')}   FPS {fps}",
                        (18, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            sparkline(frame, list(bat_hist), 18, 510, 250, 70, vmin=0, vmax=100, label="BAT history")
            sparkline(frame, list(h_hist), 288, 510, 250, 70, vmin=0, vmax=200, label="H history")
            sparkline(frame, list(tof_hist), 18, 595, 250, 70, vmin=0, vmax=200, label="TOF history")
            sparkline(frame, list(vxy_hist), 288, 595, 250, 70, vmin=0, vmax=60, label="Vxy cm/s")

            cv2.putText(frame,
                        "Keys: Q quit | A arm | T takeoff | L land (double) | O open | P close | M mission | C sdk | R rec | S snap",
                        (18, h_img - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if logger.enabled:
                cv2.circle(frame, (panel_w - 20, 28), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (panel_w - 70, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(win, frame)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if now - last_key_action < KEY_COOLDOWN_S:
                continue
            last_key_action = now

            if key in (ord("q"), ord("Q")):
                break

            elif key in (ord("a"), ord("A")):
                armed = not armed
                with reply_lock:
                    last_cmd_reply = "ARMED" if armed else "DISARMED"

            elif key in (ord("c"), ord("C")):
                sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)

            elif key in (ord("r"), ord("R")):
                if not logger.enabled:
                    logger.start()
                    with reply_lock:
                        last_cmd_reply = f"recording: ON ({logger.filename})"
                else:
                    fn = logger.filename
                    logger.stop()
                    with reply_lock:
                        last_cmd_reply = f"recording: OFF ({fn})"

            elif key in (ord("s"), ord("S")):
                fn = f"tello_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(fn, frame)
                with reply_lock:
                    last_cmd_reply = f"snapshot saved: {fn}"

            elif key in (ord("o"), ord("O")):
                if not servo.is_connected():
                    servo.auto_connect()
                ok = servo.send("OPEN")
                with reply_lock:
                    last_cmd_reply = "servo: OPEN" if ok else "servo: OPEN failed"

            elif key in (ord("p"), ord("P")):
                if not servo.is_connected():
                    servo.auto_connect()
                ok = servo.send("CLOSE")
                with reply_lock:
                    last_cmd_reply = "servo: CLOSE" if ok else "servo: CLOSE failed"

            elif key in (ord("m"), ord("M")):
                mission_pickup_drop()

            elif key in (ord("t"), ord("T")):
                if not armed:
                    with reply_lock:
                        last_cmd_reply = "blocked: DISARMED (press A)"
                    continue
                if not sdk_ok:
                    sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)
                    if not sdk_ok:
                        with reply_lock:
                            last_cmd_reply = "blocked: SDK not ready (press C)"
                        continue
                if bat is not None and bat <= BAT_CRIT:
                    with reply_lock:
                        last_cmd_reply = "blocked: battery critical"
                    continue
                if state_level == "CRIT":
                    with reply_lock:
                        last_cmd_reply = "blocked: state link critical"
                    continue

                ok = t.send_and_wait_ok("takeoff", timeout_s=8.0)
                with reply_lock:
                    last_cmd_reply = "takeoff: ok" if ok else "takeoff: no ok"

            elif key in (ord("l"), ord("L")):
                if not armed:
                    with reply_lock:
                        last_cmd_reply = "blocked: DISARMED (press A)"
                    continue

                if now - last_land_key_ts > LAND_DOUBLE_PRESS_WINDOW_S:
                    last_land_key_ts = now
                    with reply_lock:
                        last_cmd_reply = "LAND: press L again to confirm"
                    continue

                last_land_key_ts = 0.0
                ok = t.send_and_wait_ok("land", timeout_s=8.0)
                with reply_lock:
                    last_cmd_reply = "land: ok" if ok else "land: no ok"

    finally:
        running = False
        try:
            t.send("streamoff", delay=0.2)
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
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
        t.close()


if __name__ == "__main__":
    main()
