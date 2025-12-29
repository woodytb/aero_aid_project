import time

BAT_WARN = 30
BAT_CRIT = 20

STATE_WARN_AGE_S = 1.0
STATE_CRIT_AGE_S = 2.5

VIDEO_WARN_AGE_S = 1.0
VIDEO_CRIT_AGE_S = 2.5

TOF_MIN_VALID = 5
TOF_MAX_VALID = 300


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


def tof_level(tof):
    if tof is None:
        return "WARN"
    if tof < TOF_MIN_VALID or tof > TOF_MAX_VALID:
        return "WARN"
    return "OK"
