import socket
import threading
import time
import cv2
import numpy as np

TELLO_IP = "192.168.10.1"
CMD_PORT = 8889
STATE_PORT = 8890
VIDEO_PORT = 11111

LOCAL_CMD_PORT = 9000

running = True
latest_state = {}
state_lock = threading.Lock()

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


class TelloClient:
    def __init__(self):
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind(("", LOCAL_CMD_PORT))
        self.cmd_sock.settimeout(2.0)
        self.tello_addr = (TELLO_IP, CMD_PORT)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind(("", STATE_PORT))
        self.state_sock.settimeout(1.0)

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
        global latest_state
        while running:
            try:
                data, _ = self.state_sock.recvfrom(2048)
                st = parse_state(data.decode("utf-8", errors="ignore"))
                with state_lock:
                    latest_state = st
            except socket.timeout:
                continue
            except OSError:
                break

    def send(self, cmd: str, delay: float = 0.15):
        global last_sent_cmd
        try:
            self.cmd_sock.sendto(cmd.encode("utf-8"), self.tello_addr)
            print(f"[{ts()}] TELLO CMD: {cmd}")
            with reply_lock:
                last_sent_cmd = cmd
        except OSError as e:
            print(f"[{ts()}] SEND ERROR: {e}")
            with reply_lock:
                global last_cmd_reply
                last_cmd_reply = f"SEND ERROR: {e}"
        time.sleep(delay)

    def send_and_wait_ok(self, cmd: str, timeout_s: float = 1.5) -> bool:
        """
        Sends a command and waits until we receive 'ok' (or 'error') reply.
        Note: Tello replies are best-effort; if nothing comes back, returns False.
        """
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


def hud_lines(st: dict) -> list[str]:
    if not st:
        return ["No state yet (WLAN/Port 8890/Firewall check)"]

    return [
        f"BAT   : {st.get('bat','?')} %",
        f"H     : {st.get('h','?')} cm    TOF: {st.get('tof','?')} cm",
        f"TIME  : {st.get('time','?')} s   BARO: {st.get('baro','?')}",
        f"TEMP  : {st.get('templ','?')}-{st.get('temph','?')} C",
        f"ATT   : yaw {st.get('yaw','?')}  pitch {st.get('pitch','?')}  roll {st.get('roll','?')}",
        f"VEL   : vgx {st.get('vgx','?')}  vgy {st.get('vgy','?')}  vgz {st.get('vgz','?')}",
    ]


def draw_panel(img, st: dict, sent_cmd: str, last_reply: str, stream_ok: bool, sdk_ok: bool):
    h, w = img.shape[:2]
    panel_w = min(620, max(460, w // 2))

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    y = 35
    cv2.putText(img, "TELLO OPERATOR DASHBOARD", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    y += 38

    cv2.putText(img, f"SDK: {'OK' if sdk_ok else 'NOT READY'}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    y += 30

    cv2.putText(img, f"Video: {'OK' if stream_ok else 'WAITING'}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    y += 35

    cv2.putText(img, f"Sent: {sent_cmd}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    y += 28

    cv2.putText(img, f"Reply: {last_reply}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    y += 38

    for line in hud_lines(st):
        cv2.putText(img, line, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        y += 30

    y = h - 20
    cv2.putText(img, "Keys: Q quit | T takeoff | L land | S snapshot | C re-enter SDK",
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)


def open_video_capture():
    # macOS-friendly candidates
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


def main():
    global running, last_cmd_reply, last_sent_cmd

    t = TelloClient()

    # Ensure SDK mode with ack
    sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)
    if not sdk_ok:
        # try again
        time.sleep(0.3)
        sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)

    # Start stream (no guaranteed ok on some setups, but usually ok)
    t.send("streamon", delay=0.5)
    t.send("streamon", delay=0.5)

    cap, uri = open_video_capture()
    print(f"[{ts()}] Video capture URI: {uri} | opened: {cap.isOpened()}")

    win = "Tello Operator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1100, 700)

    stream_ok = False
    last_frame_time = 0.0

    try:
        while True:
            ret, frame = (False, None)
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()

            now = time.time()

            if ret and frame is not None:
                stream_ok = True
                last_frame_time = now
            else:
                if now - last_frame_time > 2.0:
                    stream_ok = False
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for video... (WLAN? streamon? Firewall?)",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            with state_lock:
                st = dict(latest_state)
            with reply_lock:
                sent_cmd = str(last_sent_cmd)
                reply = str(last_cmd_reply)

            draw_panel(frame, st, sent_cmd, reply, stream_ok, sdk_ok)
            cv2.imshow(win, frame)

            # IMPORTANT: Click the window once so it has focus, then keys work.
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            elif key in (ord("c"), ord("C")):
                sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)

            elif key in (ord("t"), ord("T")):
                # If SDK isn't ready, try to enter it again
                if not sdk_ok:
                    sdk_ok = t.send_and_wait_ok("command", timeout_s=2.0)
                # Try takeoff, show result in console + reply field
                ok = t.send_and_wait_ok("takeoff", timeout_s=6.0)
                if not ok:
                    with reply_lock:
                        last_cmd_reply = "takeoff: no ok (check battery / sensors / SDK)"

            elif key in (ord("l"), ord("L")):
                ok = t.send_and_wait_ok("land", timeout_s=6.0)
                if not ok:
                    with reply_lock:
                        last_cmd_reply = "land: no ok"

            elif key in (ord("s"), ord("S")):
                fn = f"tello_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(fn, frame)
                with reply_lock:
                    last_cmd_reply = f"Saved snapshot: {fn}"

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
        cv2.destroyAllWindows()
        t.close()


if __name__ == "__main__":
    main()
