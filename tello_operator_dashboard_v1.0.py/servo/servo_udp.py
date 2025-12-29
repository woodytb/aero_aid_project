import socket

class ServoUDP:
    def __init__(self, host, port=8890, timeout=0.2):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def auto_connect(self):
        return True

    def is_connected(self):
        # UDP hat kein echtes "connected" -> wir sagen True, solange Socket existiert
        return self.sock is not None

    def send(self, line: str):
        try:
            self.sock.sendto(line.strip().encode("utf-8"), (self.host, self.port))
            return True
        except Exception:
            return False

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None
