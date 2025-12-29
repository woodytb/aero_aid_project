import socket
import threading
import time
from utils.telemetry import parse_state, ts


class TelloClient:
    def __init__(self, tello_ip="192.168.10.1", cmd_port=8889, state_port=8890, local_cmd_port=9000):
        self.tello_addr = (tello_ip, cmd_port)

        self.latest_state = {}
        self.state_lock = threading.Lock()
        self.last_state_ts = 0.0

        self.last_cmd_reply = "—"
        self.last_sent_cmd = "—"
        self.reply_lock = threading.Lock()

        self.running = True

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind(("", local_cmd_port))
        self.cmd_sock.settimeout(2.0)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind(("", state_port))
        self.state_sock.settimeout(1.0)

        self.cmd_thread = threading.Thread(target=self._cmd_recv_loop, daemon=True)
        self.state_thread = threading.Thread(target=self._state_recv_loop, daemon=True)
        self.cmd_thread.start()
        self.state_thread.start()

    def _cmd_recv_loop(self):
        while self.running:
            try:
                data, _ = self.cmd_sock.recvfrom(1024)
                msg = data.decode("utf-8", errors="ignore").strip()
                print(f"[{ts()}] TELLO REPLY: {msg}")
                with self.reply_lock:
                    self.last_cmd_reply = msg
            except socket.timeout:
                continue
            except OSError:
                break

    def _state_recv_loop(self):
        while self.running:
            try:
                data, _ = self.state_sock.recvfrom(2048)
                st = parse_state(data.decode("utf-8", errors="ignore"))
                with self.state_lock:
                    self.latest_state = st
                self.last_state_ts = time.time()
            except socket.timeout:
                continue
            except OSError:
                break

    def send(self, cmd: str, delay: float = 0.12):
        try:
            self.cmd_sock.sendto(cmd.encode("utf-8"), self.tello_addr)
            print(f"[{ts()}] TELLO CMD: {cmd}")
            with self.reply_lock:
                self.last_sent_cmd = cmd
        except OSError as e:
            print(f"[{ts()}] SEND ERROR: {e}")
            with self.reply_lock:
                self.last_cmd_reply = f"SEND ERROR: {e}"
        time.sleep(delay)

    def send_and_wait_ok(self, cmd: str, timeout_s: float = 2.0) -> bool:
        with self.reply_lock:
            self.last_cmd_reply = "—"
        self.send(cmd, delay=0.05)

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self.reply_lock:
                r = str(self.last_cmd_reply).lower()
            if r == "ok":
                return True
            if r.startswith("error"):
                return False
            time.sleep(0.05)
        return False

    def get_state_copy(self):
        with self.state_lock:
            return dict(self.latest_state)

    def get_cmd_status(self):
        with self.reply_lock:
            return str(self.last_sent_cmd), str(self.last_cmd_reply)

    def close(self):
        self.running = False
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
        while self.client.running:
            if self.enabled:
                self.client.send("command", delay=0.05)
            time.sleep(5.0)
