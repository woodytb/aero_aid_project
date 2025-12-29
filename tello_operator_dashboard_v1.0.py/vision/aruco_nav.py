import time
import threading
import numpy as np
import cv2

try:
    import cv2.aruco as aruco
    ARUCO_AVAILABLE = True
except Exception:
    aruco = None
    ARUCO_AVAILABLE = False


ARUCO_MARKER_SIZE_CM = 12.0
CAMERA_FOV_DEG = 82.0
ARUCO_DESIRED_DIST_CM = 90.0
ARUCO_DIST_TOL_CM = 20.0
ARUCO_CENTER_TOL_PX = 45

ARUCO_YAW_STEP_DEG = 10
ARUCO_MOVE_STEP_CM = 20
ARUCO_VERT_STEP_CM = 20
ARUCO_SEARCH_ROT_DEG = 25
ARUCO_LOOP_DELAY_S = 0.12
ARUCO_GOTO_TIMEOUT_S = 40.0


def focal_px_from_fov(width_px: int, fov_deg: float) -> float:
    fov = np.deg2rad(max(1.0, float(fov_deg)))
    return (width_px / 2.0) / np.tan(fov / 2.0)


def detect_aruco(frame_bgr, dict_id=None):
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
        c = corners[i].reshape(-1, 2)
        cx = float(np.mean(c[:, 0]))
        cy = float(np.mean(c[:, 1]))
        side = 0.0
        for k in range(4):
            p1 = c[k]
            p2 = c[(k + 1) % 4]
            side += float(np.linalg.norm(p2 - p1))
        side /= 4.0
        out.append({"id": int(mid), "center": (cx, cy), "size_px": side, "corners": c})
    return out


class MissionRunner:
    def __init__(self, tello, get_frame_fn, get_stream_ok_fn, is_armed_fn, ensure_sdk_fn, set_status_fn):
        self.tello = tello
        self.get_frame = get_frame_fn
        self.get_stream_ok = get_stream_ok_fn
        self.is_armed = is_armed_fn
        self.ensure_sdk = ensure_sdk_fn
        self.set_status = set_status_fn

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._job = None
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
        self.set_status("mission: ABORT requested")

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
            return self._abort

    def _loop(self):
        while True:
            with self._lock:
                while self._job is None:
                    self._cv.wait(timeout=0.25)
                job = self._job
                self._job = None
                self._busy = job is not None

            if job is None:
                continue

            try:
                if job[0] == "goto":
                    self._do_goto(job[1])
                else:
                    for mid in job[1]:
                        if self._should_abort():
                            break
                        self._do_goto(mid)
                        time.sleep(0.5)
            finally:
                with self._lock:
                    self._busy = False

    def _do_goto(self, target_id: int):
        if not ARUCO_AVAILABLE:
            self.set_status("aruco: NOT AVAILABLE (opencv-contrib-python)")
            return
        if not self.is_armed():
            self.set_status("aruco goto blocked: DISARMED")
            return
        if not self.ensure_sdk():
            self.set_status("aruco goto blocked: SDK not ready")
            return
        if not self.get_stream_ok():
            self.set_status("aruco goto blocked: NO VIDEO")
            return

        self.set_status(f"aruco goto: searching ID {target_id}...")
        t0 = time.time()
        last_search = 0.0
        found_once = False

        while time.time() - t0 < ARUCO_GOTO_TIMEOUT_S and not self._should_abort():
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            dets = detect_aruco(frame)
            target = next((d for d in dets if d["id"] == target_id), None)

            if target is None:
                if time.time() - last_search > 1.0:
                    self.tello.send_and_wait_ok(f"cw {ARUCO_SEARCH_ROT_DEG}", timeout_s=4.0)
                    last_search = time.time()
                time.sleep(ARUCO_LOOP_DELAY_S)
                continue

            found_once = True
            cx, cy = target["center"]
            size_px = max(1.0, float(target["size_px"]))

            fpx = focal_px_from_fov(w, CAMERA_FOV_DEG)
            dist_cm = (ARUCO_MARKER_SIZE_CM * fpx) / size_px

            dx = cx - (w / 2.0)
            dy = cy - (h / 2.0)

            yaw_ok = abs(dx) <= ARUCO_CENTER_TOL_PX
            vert_ok = abs(dy) <= ARUCO_CENTER_TOL_PX
            dist_ok = abs(dist_cm - ARUCO_DESIRED_DIST_CM) <= ARUCO_DIST_TOL_CM

            if yaw_ok and vert_ok and dist_ok:
                self.set_status(f"aruco goto: REACHED ID {target_id}")
                return

            # adjust yaw
            if not yaw_ok and not self._should_abort():
                self.tello.send_and_wait_ok(("ccw" if dx < 0 else "cw") + f" {ARUCO_YAW_STEP_DEG}", timeout_s=3.0)

            # adjust vertical
            if not vert_ok and not self._should_abort():
                self.tello.send_and_wait_ok(("up" if dy < 0 else "down") + f" {ARUCO_VERT_STEP_CM}", timeout_s=3.0)

            # adjust distance
            if not dist_ok and not self._should_abort():
                self.tello.send_and_wait_ok(("forward" if dist_cm > ARUCO_DESIRED_DIST_CM else "back") + f" {ARUCO_MOVE_STEP_CM}", timeout_s=4.0)

            time.sleep(ARUCO_LOOP_DELAY_S)

        if self._should_abort():
            self.set_status(f"aruco goto: ABORTED (ID {target_id})")
        else:
            self.set_status(f"aruco goto: TIMEOUT/NOT FOUND (ID {target_id})" if not found_once else f"aruco goto: TIMEOUT (ID {target_id})")
