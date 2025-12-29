import serial
from serial.tools import list_ports


class ServoSerial:
    def __init__(self, baud=115200):
        self.ser = None
        self.port = None
        self.baud = baud

    def auto_connect(self):
        ports = list(list_ports.comports())
        candidates = []
        for p in ports:
            dev = p.device
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
