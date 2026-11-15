"""Synthetic vibration generator for a rotating machine.

All signals are SIMULATED from simple physical ideas, not measured on a real machine.
Sampling: FS Hz. A 1-second window = FS samples (like an accelerometer on an ESP32).
"""
import numpy as np

FS = 1024          # samples per second (power of two, so FFT bins are exactly 1 Hz)
WINDOW = 1024      # samples per window (1 second)
RPM_HZ = 50.0      # shaft speed (3000 rpm) -> "1x" frequency

STATES = ["healthy", "imbalance", "bearing_fault", "looseness"]

AR_NAMES = {
    "healthy": "سليم",
    "imbalance": "عدم اتزان",
    "bearing_fault": "عطل محمل",
    "looseness": "ارتخاء",
}


def make_window(state: str, severity: float = 0.6, rng=None, shaft_hz: float = RPM_HZ):
    """Return one window of acceleration (g) for the given state.

    severity in [0, 1]: how strong the fault is (ignored for healthy).
    """
    if state not in STATES:
        raise ValueError(f"unknown state: {state}")
    rng = rng or np.random.default_rng()
    t = np.arange(WINDOW) / FS
    phase = rng.uniform(0, 2 * np.pi)
    s = float(np.clip(severity, 0, 1))

    # baseline: shaft frequency + a little noise
    x = 0.25 * np.sin(2 * np.pi * shaft_hz * t + phase)
    x += rng.normal(0, 0.05, WINDOW)

    if state == "imbalance":
        # much stronger 1x component
        x += (0.3 + 1.6 * s) * np.sin(2 * np.pi * shaft_hz * t + phase)
    elif state == "bearing_fault":
        # periodic impacts that ring a high frequency (~300 Hz) and decay quickly
        impact_hz = 87.0 + rng.normal(0, 1.0)  # fault repetition rate
        ring = 300.0
        n_imp = int(impact_hz)
        for k in range(n_imp):
            t0 = (k / impact_hz) + rng.normal(0, 0.0005)
            idx = max(0, int(t0 * FS))
            if idx >= WINDOW - 1:
                continue
            tail = np.arange(WINDOW - idx) / FS
            x[idx:] += (0.15 + 1.4 * s) * np.exp(-tail * 700) * np.sin(2 * np.pi * ring * tail)
        x += rng.normal(0, 0.03 + 0.1 * s, WINDOW)
    elif state == "looseness":
        # harmonics at 2x, 3x, 4x (and a half order) with some clipping
        for h, a in [(2, 0.7), (3, 0.5), (4, 0.35), (0.5, 0.3)]:
            x += a * (0.2 + 1.2 * s) * np.sin(2 * np.pi * shaft_hz * h * t + rng.uniform(0, 6.28))
    return x.astype(np.float32)


def make_dataset(n_per_class=400, seed=0):
    """Many windows with random severity, shaft speed jitter, and noise."""
    rng = np.random.default_rng(seed)
    X, y, sevs = [], [], []
    for label, state in enumerate(STATES):
        for _ in range(n_per_class):
            sev = rng.uniform(0.05, 1.0) if state != "healthy" else 0.0
            shaft = RPM_HZ + rng.normal(0, 0.8)
            w = make_window(state, sev, rng, shaft)
            # sensor/mount variation: random gain and extra noise, so classes overlap a bit
            w = w * rng.uniform(0.8, 1.25) + rng.normal(0, rng.uniform(0, 0.12), WINDOW)
            X.append(w.astype(np.float32))
            y.append(label)
            sevs.append(sev)
    return np.stack(X), np.array(y), np.array(sevs)
