"""Train the small classifier on simulated data and save model + metrics."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from .features import FEATURE_NAMES, extract_many
from .simulator import STATES, make_dataset

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main(n_per_class=400, seed=0):
    X, y, sev = make_dataset(n_per_class, seed)
    F = extract_many(X)
    Ftr, Fte, ytr, yte, str_, ste = train_test_split(F, y, sev, test_size=0.25, stratify=y, random_state=seed)
    clf = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=seed)
    clf.fit(Ftr, ytr)
    reg = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=seed).fit(Ftr, str_)
    mae = float(np.mean(np.abs(reg.predict(Fte) - ste)))
    pred = clf.predict(Fte)
    acc = float(accuracy_score(yte, pred))
    cm = confusion_matrix(yte, pred).tolist()
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "model.joblib")
    joblib.dump(reg, MODEL_DIR / "severity.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps({
        "accuracy": acc, "severity_mae": mae, "states": STATES, "confusion_matrix": cm,
        "features": FEATURE_NAMES, "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "note": "Simulated data only. Not validated on a real machine.",
    }, indent=2))
    print(f"accuracy on held-out simulated windows: {acc:.3f}; severity MAE: {mae:.3f}")
    return acc


if __name__ == "__main__":
    main()
