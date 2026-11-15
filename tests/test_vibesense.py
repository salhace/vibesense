import numpy as np
from vibesense.features import FEATURE_NAMES, extract
from vibesense.simulator import STATES, WINDOW, make_dataset, make_window


def test_window_shape_and_states():
    rng = np.random.default_rng(0)
    for s in STATES:
        assert make_window(s, 0.5, rng).shape == (WINDOW,)


def test_unknown_state_rejected():
    import pytest
    with pytest.raises(ValueError):
        make_window("nope")


def test_imbalance_has_stronger_shaft_component():
    rng = np.random.default_rng(1)
    h = extract(make_window("healthy", 0, rng))
    i = extract(make_window("imbalance", 0.8, rng))
    assert i[FEATURE_NAMES.index("band_30_70")] > 2 * h[FEATURE_NAMES.index("band_30_70")]


def test_bearing_fault_raises_kurtosis_or_high_band():
    rng = np.random.default_rng(2)
    h = extract(make_window("healthy", 0, rng))
    b = extract(make_window("bearing_fault", 0.8, rng))
    hi = FEATURE_NAMES.index("band_230_380")
    assert b[hi] > 3 * h[hi]


def test_dataset_and_model_accuracy():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from vibesense.features import extract_many
    X, y, sev = make_dataset(120, seed=3)
    F = extract_many(X)
    Ftr, Fte, ytr, yte = train_test_split(F, y, test_size=0.3, stratify=y, random_state=0)
    acc = RandomForestClassifier(n_estimators=40, random_state=0).fit(Ftr, ytr).score(Fte, yte)
    assert acc > 0.9
