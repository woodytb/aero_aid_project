import csv
import os
import time


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
