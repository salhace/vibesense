"""Turn a raw vibration window into a few numbers a small model can use."""
import numpy as np
from .simulator import FS

# frequency bands (Hz) chosen around the shaft speed and its multiples
BANDS = [(0, 30), (30, 70), (70, 130), (130, 230), (230, 380), (380, 500)]
FEATURE_NAMES = (
    ["rms", "peak", "crest", "kurtosis"]
    + [f"band_{a}_{b}" for a, b in BANDS]
    + ["peak_freq"]
)


def spectrum(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w)) * 2 / w.sum()
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    return freqs, mag


def extract(x):
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    crest = peak / (rms + 1e-9)
    c = x - x.mean()
    kurt = float(np.mean(c ** 4) / (np.mean(c ** 2) ** 2 + 1e-12))
    f, m = spectrum(x)
    bands = []
    for a, b in BANDS:
        sel = (f >= a) & (f < b)
        bands.append(float(np.sqrt(np.sum(m[sel] ** 2))))
    pf = float(f[1:][np.argmax(m[1:])])
    return np.array([rms, peak, crest, kurt, *bands, pf], dtype=np.float64)


def extract_many(X):
    return np.stack([extract(x) for x in X])
