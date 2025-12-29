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

# NEW (ArUco):
try:
    import cv2.aruco as aruco
    ARUCO_AVAILABLE = True
except Exception:
    aruco = None
    ARUCO_AVAILABLE = False

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


# ----------------- NEW: ArUco + Missions -----------------
ARUCO_MARKER_SIZE_CM = 12.0          # <- anpassen an deine Marker-Kantenlänge (cm)
CAMERA_FOV_DEG = 82.0                # grobe Annahme für Tello
ARUCO_DESIRED_DIST_CM = 90.0         # gewünschter Abstand zum Marker (cm)
ARUCO_DIST_TOL_CM = 20.0
ARUCO_CENTER_TOL_PX = 45

ARUCO_YAW_STEP_DEG = 10
ARUCO_MOVE_STEP_CM = 20
ARUCO_VERT_STEP_CM = 20
ARUCO_SEARCH_ROT_DEG = 25
ARUCO_LOOP_DELAY_S = 0.12
ARUCO_GOTO_TIMEOUT_S = 40.0

DEFAULT_MISSIONS = [
    ("Mission 0: IDs 0-1-2", [0, 1, 2]),
    ("Mission 1: IDs 2-1-0", [2, 1, 0]),
    ("Mission 2: IDs 0-2", [0, 2]),
]


def _focal_px_from_fov(width_px: int, fov_deg: float) -> float:
    fov = np.deg2rad(max(1.0, float(fov_deg)))
    return (width_px / 2.0) / np.tan(fov / 2.0)


def detect_aruco(frame_bgr, dict_id=None):
    """
    Returns:
      detections: list of dicts {id, center(x,y), size_px, corners}
    """
    if not ARUCO_AVAILABLE or frame_bgr is None:
        return []

    if dict_id is None:
        dict_id = aruco.DICT_4X4_50

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    ar_dict = aruco.getPredefinedDictionary(dict_id)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(ar_dict, params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return []

    out = []
    ids = ids.flatten().tolist()
    for i, mid in enumerate(ids):
        c = corners[i].reshape(-1, 2)  # 4x2
        cx = float(np.mean(c[:, 0]))
        cy = float(np.mean(c[:, 1]))
        # average side length in px
        side = 0.0
        for k in range(4):
            p1 = c[k]
            p2 = c[(k + 1) % 4]
            side += float(np.linalg.norm(p2 - p1))
        side /= 4.0
        out.append({
            "id": int(mid),
            "center": (cx, cy),
            "size_px": side,
            "corners": c
        })
    return out


class MissionRunner:
    """
    Runs goto-aruco / mission sequences in background so UI stays alive.
    """
    def __init__(self, tello: TelloClient, get_frame_fn, get_stream_ok_fn, is_armed_fn, ensure_sdk_fn):
        self.tello = tello
        self.get_frame = get_frame_fn
        self.get_stream_ok = get_stream_ok_fn
        self.is_armed = is_armed_fn
        self.ensure_sdk = ensure_sdk_fn

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._job = None   # ("goto", id) or ("mission", [ids])
        self._abort = False
        self._busy = False

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def abort(self):
        with self._lock:
            self._abort = True
            self._job = None
            self._cv.notify_all()
        with reply_lock:
            global last_cmd_reply
            last_cmd_reply = "mission: ABORT requested"

    def goto_id(self, aruco_id: int):
        with self._lock:
            self._abort = False
            self._job = ("goto", int(aruco_id))
            self._cv.notify_all()

    def run_mission(self, ids_list):
        with self._lock:
            self._abort = False
            self._job = ("mission", [int(x) for x in ids_list])
            self._cv.notify_all()

    def _should_abort(self):
        with self._lock:
            return bool(self._abort) or not running

    def _loop(self):
        while running:
            with self._lock:
                while running and self._job is None:
                    self._cv.wait(timeout=0.25)
                job = self._job
                self._job = None
                self._busy = job is not None

            if not running:
                break
            if job is None:
                continue

            try:
                kind = job[0]
                if kind == "goto":
                    self._do_goto(job[1])
                elif kind == "mission":
                    for mid in job[1]:
                        if self._should_abort():
                            break
                        self._do_goto(mid)
                        time.sleep(0.5)
            finally:
                with self._lock:
                    self._busy = False

    def _do_goto(self, target_id: int):
        global last_cmd_reply

        if not ARUCO_AVAILABLE:
            with reply_lock:
                last_cmd_reply = "aruco: NOT AVAILABLE (install opencv-contrib-python)"
            return

        if not self.is_armed():
            with reply_lock:
                last_cmd_reply = "aruco goto blocked: DISARMED"
            return

        if not self.ensure_sdk():
            with reply_lock:
                last_cmd_reply = "aruco goto blocked: SDK not ready"
            return

        if not self.get_stream_ok():
            with reply_lock:
                last_cmd_reply = "aruco goto blocked: NO VIDEO"
            return

        with reply_lock:
            last_cmd_reply = f"aruco goto: searching ID {target_id}..."

        t0 = time.time()
        last_seen = 0.0
        found_once = False

        while time.time() - t0 < ARUCO_GOTO_TIMEOUT_S and not self._should_abort():
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            detections = detect_aruco(frame)

            target = None
            for d in detections:
                if d["id"] == target_id:
                    target = d
                    break

            if target is None:
                # search behavior: rotate occasionally
                if time.time() - last_seen > 1.0:
                    self.tello.send_and_wait_ok(f"cw {ARUCO_SEARCH_ROT_DEG}", timeout_s=4.0)
                    last_seen = time.time()
                with reply_lock:
                    last_cmd_reply = f"aruco goto: searching ID {target_id}..."
                time.sleep(ARUCO_LOOP_DELAY_S)
                continue

            found_once = True
            last_seen = time.time()

            cx, cy = target["center"]
            size_px = max(1.0, float(target["size_px"]))

            fpx = _focal_px_from_fov(w, CAMERA_FOV_DEG)
            dist_cm = (ARUCO_MARKER_SIZE_CM * fpx) / size_px

            dx = cx - (w / 2.0)
            dy = cy - (h / 2.0)

            yaw_ok = abs(dx) <= ARUCO_CENTER_TOL_PX
            vert_ok = abs(dy) <= ARUCO_CENTER_TOL_PX
            dist_ok = abs(dist_cm - ARUCO_DESIRED_DIST_CM) <= ARUCO_DIST_TOL_CM

            with reply_lock:
                last_cmd_reply = (
                    f"aruco ID {target_id}  dist~{dist_cm:.0f}cm  "
                    f"dx={dx:.0f}px dy={dy:.0f}px  "
                    f"{'OK' if (yaw_ok and vert_ok and dist_ok) else 'adjusting...'}"
                )

            if yaw_ok and vert_ok and dist_ok:
                with reply_lock:
                    last_cmd_reply = f"aruco goto: REACHED ID {target_id}"
                return

            # adjust yaw
            if not yaw_ok and not self._should_abort():
                if dx < 0:
                    self.tello.send_and_wait_ok(f"ccw {ARUCO_YAW_STEP_DEG}", timeout_s=3.0)
                else:
                    self.tello.send_and_wait_ok(f"cw {ARUCO_YAW_STEP_DEG}", timeout_s=3.0)

            # adjust vertical
            if not vert_ok and not self._should_abort():
                if dy < 0:
                    self.tello.send_and_wait_ok(f"up {ARUCO_VERT_STEP_CM}", timeout_s=3.0)
                else:
                    self.tello.send_and_wait_ok(f"down {ARUCO_VERT_STEP_CM}", timeout_s=3.0)

            # adjust distance
            if not dist_ok and not self._should_abort():
                if dist_cm > ARUCO_DESIRED_DIST_CM:
                    self.tello.send_and_wait_ok(f"forward {ARUCO_MOVE_STEP_CM}", timeout_s=4.0)
                else:
                    self.tello.send_and_wait_ok(f"back {ARUCO_MOVE_STEP_CM}", timeout_s=4.0)

            time.sleep(ARUCO_LOOP_DELAY_S)

        if self._should_abort():
            with reply_lock:
                last_cmd_reply = f"aruco goto: ABORTED (ID {target_id})"
        else:
            with reply_lock:
                last_cmd_reply = f"aruco goto: TIMEOUT (ID {target_id})" if found_once else f"aruco goto: NOT FOUND (ID {target_id})"


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

    # -------- NEW: window first (dashboard always opens), even if stream not connected --------
    win = "Tello Operator PRO + SERVO + ARUCO"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    # shared latest frame for mission runner
    frame_lock = threading.Lock()
    latest_frame_bgr = None

    # Init SDK + video (non-fatal)
    t.send("command", delay=0.3)
    sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)
    t.send("streamon", delay=0.3)

    cap = None
    uri = "none"
    cap, uri = open_video_capture()
    print(f"[{ts()}] Video capture URI: {uri} | opened: {cap.isOpened()}")

    keepalive = KeepAlive(t)
    keepalive.enabled = True

    stream_ok = False
    last_frame_ts = 0.0
    fps_hist = deque([], maxlen=60)
    prev_frame_ts = None

    # NEW: video retry timer
    last_video_retry_ts = 0.0
    VIDEO_RETRY_EVERY_S = 3.0

    # -------- NEW: missions + selected aruco id ----------
    missions = list(DEFAULT_MISSIONS)
    mission_idx = 0
    selected_aruco_id = 0

    def ensure_sdk():
        nonlocal sdk_ok
        if sdk_ok:
            return True
        sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)
        return sdk_ok

    def get_stream_ok():
        return bool(stream_ok)

    def get_frame_copy():
        with frame_lock:
            if latest_frame_bgr is None:
                return None
            return latest_frame_bgr.copy()

    def is_armed():
        return bool(armed)

    runner = MissionRunner(
        t,
        get_frame_fn=get_frame_copy,
        get_stream_ok_fn=get_stream_ok,
        is_armed_fn=is_armed,
        ensure_sdk_fn=ensure_sdk
    )

    def start_or_retry_video():
        nonlocal cap, uri, last_video_retry_ts
        now2 = time.time()
        if now2 - last_video_retry_ts < VIDEO_RETRY_EVERY_S:
            return
        last_video_retry_ts = now2
        try:
            t.send("streamon", delay=0.2)
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        cap, uri = open_video_capture()
        print(f"[{ts()}] Video retry URI: {uri} | opened: {cap.isOpened()}")

    def mission_pickup_drop():
        """
        land -> close -> takeoff -> right 100 -> land -> open
        """
        nonlocal servo_status, servo_connected

        if not armed:
            with reply_lock:
                global last_cmd_reply
                last_cmd_reply = "mission blocked: DISARMED"
            return
        if not ensure_sdk():
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

        ok = t.send_and_wait_ok("land", timeout_s=8.0)
        if not ok:
            with reply_lock:
                last_cmd_reply = "mission: land failed"
            return

        time.sleep(1.0)

        if servo.send("CLOSE"):
            with reply_lock:
                last_cmd_reply = "mission: CLOSE sent"
        else:
            with reply_lock:
                last_cmd_reply = "mission: CLOSE failed"
            return

        time.sleep(1.0)

        ok = t.send_and_wait_ok("takeoff", timeout_s=8.0)
        if not ok:
            with reply_lock:
                last_cmd_reply = "mission: takeoff failed"
            return

        ok = t.send_and_wait_ok("right 100", timeout_s=5.0)
        if not ok:
            with reply_lock:
                last_cmd_reply = "mission: right 100 failed"
            return

        ok = t.send_and_wait_ok("land", timeout_s=8.0)
        if not ok:
            with reply_lock:
                last_cmd_reply = "mission: land2 failed"
            return

        time.sleep(1.0)

        if servo.send("OPEN"):
            with reply_lock:
                last_cmd_reply = "mission: OPEN sent"
        else:
            with reply_lock:
                last_cmd_reply = "mission: OPEN failed"

    try:
        while True:
            now = time.time()

            # Video read (non-fatal)
            ret, frame = (False, None)
            if cap is not None and cap.isOpened():
                try:
                    ret, frame = cap.read()
                except Exception:
                    ret, frame = (False, None)

            if ret and frame is not None:
                stream_ok = True
                last_frame_ts = now
                if prev_frame_ts is not None:
                    dt = now - prev_frame_ts
                    if dt > 0:
                        fps_hist.append(1.0 / dt)
                prev_frame_ts = now

                with frame_lock:
                    latest_frame_bgr = frame
            else:
                # If video is stale, attempt periodic reconnect (dashboard still runs!)
                if now - last_frame_ts > 2.0:
                    stream_ok = False
                    start_or_retry_video()

                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "Dashboard läuft - kein Video-Stream.",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, "Tipps: WLAN zur Tello, 'C' SDK, 'V' Stream Retry",
                            (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
                if not ARUCO_AVAILABLE:
                    cv2.putText(frame, "Aruco: opencv-contrib-python fehlt!",
                                (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

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

            # Optional: draw aruco detections (only for visualization)
            if stream_ok and ARUCO_AVAILABLE:
                dets = detect_aruco(frame)
                for d in dets[:10]:
                    c = d["corners"].astype(np.int32)
                    cv2.polylines(frame, [c], True, (0, 255, 0), 2)
                    cx, cy = d["center"]
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID {d['id']}", (int(cx) + 8, int(cy) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Panel
            h_img, w_img = frame.shape[:2]
            panel_w = 560
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (panel_w, h_img), (0, 0, 0), -1)
            frame[:] = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

            cv2.putText(frame, "TELLO OPERATOR PRO + SERVO + ARUCO", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

            draw_badge(frame, "MODE", "ARMED" if armed else "DISARMED", "OK" if armed else "WARN", 18, 55)
            draw_badge(frame, "SDK", "OK" if sdk_ok else "NOT READY", "OK" if sdk_ok else "WARN", 18, 95)

            draw_badge(frame, "STATE", f"{state_level} ({state_age:.1f}s)", state_level, 18, 135)
            draw_badge(frame, "VIDEO", f"{video_level} ({video_age:.1f}s)", video_level, 18, 175)

            draw_badge(frame, "BAT", f"{bat if bat is not None else '?'}%", bat_level, 18, 215)
            draw_badge(frame, "H", f"{h if h is not None else '?'} cm", "OK" if h is not None else "WARN", 18, 255)
            draw_badge(frame, "TOF", f"{tof if tof is not None else '?'} cm", tof_level, 18, 295)

            servo_level = "OK" if servo.is_connected() else "WARN"
            draw_badge(frame, "SERVO", "OK" if servo.is_connected() else "NOT CONNECTED", servo_level, 18, 335)

            # NEW: ArUco / mission info
            ar_level = "OK" if ARUCO_AVAILABLE else "WARN"
            draw_badge(frame, "ARUCO", "OK" if ARUCO_AVAILABLE else "NO MODULE", ar_level, 288, 335)

            mission_name, mission_ids = missions[mission_idx]
            draw_badge(frame, "MISSION", f"{mission_idx}: {mission_ids}", "OK", 18, 375)
            draw_badge(frame, "TARGET", f"ID {selected_aruco_id}", "OK", 288, 375)

            with reply_lock:
                sent = str(last_sent_cmd)
                rep = str(last_cmd_reply)

            cv2.putText(frame, f"Sent : {sent}", (18, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Reply: {rep}", (18, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"ATT  yaw {st.get('yaw','?')}  pitch {st.get('pitch','?')}  roll {st.get('roll','?')}",
                        (18, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"VEL  vgx {st.get('vgx','?')}  vgy {st.get('vgy','?')}  vgz {st.get('vgz','?')}   FPS {fps}",
                        (18, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            sparkline(frame, list(bat_hist), 18, 550, 250, 70, vmin=0, vmax=100, label="BAT history")
            sparkline(frame, list(h_hist), 288, 550, 250, 70, vmin=0, vmax=200, label="H history")
            sparkline(frame, list(tof_hist), 18, 635, 250, 70, vmin=0, vmax=200, label="TOF history")
            sparkline(frame, list(vxy_hist), 288, 635, 250, 70, vmin=0, vmax=60, label="Vxy cm/s")

            if runner.busy:
                cv2.putText(frame, "MISSION RUNNING (press X to ABORT)", (18, 615),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            if logger.enabled:
                cv2.circle(frame, (panel_w - 20, 28), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (panel_w - 70, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame,
                        "Keys: Q quit | A arm | T takeoff | L land (double) | O open | P close | U servo-mission | "
                        "N next-mission | M run-mission | 0-9 set target | G goto ID | X abort | C sdk | V video retry | R rec | S snap",
                        (18, h_img - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

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

            # allow abort even while busy
            if key in (ord("x"), ord("X")):
                runner.abort()
                continue

            elif key in (ord("a"), ord("A")):
                armed = not armed
                with reply_lock:
                    last_cmd_reply = "ARMED" if armed else "DISARMED"

            elif key in (ord("c"), ord("C")):
                sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)

            elif key in (ord("v"), ord("V")):
                start_or_retry_video()
                with reply_lock:
                    last_cmd_reply = f"video: retry ({uri})"

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

            # Keep your original servo mission, moved to U (so M is free for ArUco missions)
            elif key in (ord("u"), ord("U")):
                if runner.busy:
                    with reply_lock:
                        last_cmd_reply = "blocked: mission runner busy (abort with X)"
                else:
                    mission_pickup_drop()

            # NEW: cycle missions
            elif key in (ord("n"), ord("N")):
                mission_idx = (mission_idx + 1) % len(missions)
                with reply_lock:
                    last_cmd_reply = f"mission selected: {missions[mission_idx][0]}"

            # NEW: run selected mission (Aruco IDs sequence)
            elif key in (ord("m"), ord("M")):
                if runner.busy:
                    with reply_lock:
                        last_cmd_reply = "mission already running (abort with X)"
                    continue
                _, ids_list = missions[mission_idx]
                runner.run_mission(ids_list)

            # NEW: goto selected aruco id
            elif key in (ord("g"), ord("G")):
                if runner.busy:
                    with reply_lock:
                        last_cmd_reply = "busy: abort with X"
                    continue
                runner.goto_id(selected_aruco_id)

            # NEW: set selected aruco id via number keys 0-9
            elif ord("0") <= key <= ord("9"):
                selected_aruco_id = int(chr(key))
                with reply_lock:
                    last_cmd_reply = f"target set: ArUco ID {selected_aruco_id}"

            elif key in (ord("t"), ord("T")):
                if not armed:
                    with reply_lock:
                        last_cmd_reply = "blocked: DISARMED (press A)"
                    continue
                if not ensure_sdk():
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
            runner.abort()
        except Exception:
            pass
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
