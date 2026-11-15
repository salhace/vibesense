"""The C code that would run on the ESP32 must agree with the Python reference.

Skipped automatically if gcc is not installed.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    out = tmp_path_factory.mktemp("fw") / "host_test"
    subprocess.run(["gcc", "-O2", "-Wall", f"-I{ROOT / 'firmware' / 'wokwi'}",
                    str(ROOT / "firmware" / "host" / "host_test.c"), "-o", str(out), "-lm"], check=True)
    return out


def test_c_matches_python(binary):
    ref = json.loads((ROOT / "firmware" / "reference.json").read_text())
    text = "\n".join(" ".join(f"{v:.9g}" for v in w) for w in ref["windows"]) + "\n"
    res = subprocess.run([str(binary), "features"], input=text, capture_output=True, text=True, check=True)
    rows = np.array([[float(v) for v in line.split()] for line in res.stdout.strip().splitlines()])
    feats, proba, sev = rows[:, :11], rows[:, 11:15], rows[:, 15]
    assert len(rows) == len(ref["windows"])
    np.testing.assert_allclose(feats, np.array(ref["features"]), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(proba, np.array(ref["clf_proba"]), atol=0.05)
    np.testing.assert_allclose(sev, np.array(ref["sev"]), atol=0.05)
    assert (proba.argmax(1) == np.array(ref["clf_proba"]).argmax(1)).mean() >= 0.95


def test_c_simulator_is_classified_correctly(binary):
    out = subprocess.run([str(binary), "selftest"], capture_output=True, text=True, check=True).stdout
    acc = float(out.split("accuracy")[1].split()[0])
    assert acc > 0.9
