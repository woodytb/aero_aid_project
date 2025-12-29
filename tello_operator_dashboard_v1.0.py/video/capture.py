import time
import cv2


class VideoManager:
    def __init__(self, video_port=11111):
        self.video_port = video_port
        self.cap = None
        self.uri = "none"
        self.last_frame_ts = 0.0
        self.stream_ok = False
        self.last_retry_ts = 0.0

    def open(self):
        candidates = [
            ("udp://@:11111", cv2.CAP_FFMPEG),
            (f"udp://0.0.0.0:{self.video_port}", cv2.CAP_FFMPEG),
            (f"udp://127.0.0.1:{self.video_port}", cv2.CAP_FFMPEG),
        ]
        for uri, backend in candidates:
            cap = cv2.VideoCapture(uri, backend)
            time.sleep(0.25)
            if cap.isOpened():
                self.cap = cap
                self.uri = uri
                return True
            cap.release()
        self.cap = cv2.VideoCapture()
        self.uri = "none"
        return False

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            self.stream_ok = False
            return False, None
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.stream_ok = True
            self.last_frame_ts = time.time()
            return True, frame
        # stale
        if time.time() - self.last_frame_ts > 2.0:
            self.stream_ok = False
        return False, None

    def retry_if_needed(self, every_s=3.0):
        now = time.time()
        if now - self.last_retry_ts < every_s:
            return False
        self.last_retry_ts = now
        self.release()
        return self.open()

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
