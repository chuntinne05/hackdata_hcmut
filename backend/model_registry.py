"""
Nested lazy registry: registry[bundle_id] -> ModelBundle (scalers + xgb + rf + lstm).
Artifacts live under backend/models/<bundle_id>/ with bundle-specific filenames.
"""
from __future__ import annotations

import glob
import os
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb


FEATURE_COLS = [
    "demand_count",
    "total_revenue",
    "pickup_hour",
    "dayofweek",
    "is_weekend",
]
TIME_STEPS = 24
# Excluded from discovery (UI / WS bundle list); artifacts may remain on disk.
_EXCLUDED_BUNDLE_IDS = frozenset({"baseline"})


class TaxiLSTM(nn.Module):
    """Matches bigdata.ipynb TaxiLSTM architecture."""

    def __init__(self, input_size: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]
        return self.fc(out)


def _first_glob(dir_path: Path, pattern: str) -> Optional[Path]:
    matches = sorted(glob.glob(str(dir_path / pattern)))
    if not matches:
        return None
    return Path(matches[0])


@dataclass
class ModelBundle:
    bundle_id: str
    bundle_dir: Path
    feature_scaler: Any
    target_scaler: Any
    xgb_model: xgb.XGBRegressor
    rf_model: Any
    lstm: nn.Module
    input_size: int

    @classmethod
    def load(cls, bundle_dir: Path) -> "ModelBundle":
        bundle_id = bundle_dir.name
        fs = _first_glob(bundle_dir, "feature_scaler*.pkl")
        ts = _first_glob(bundle_dir, "target_scaler*.pkl")
        rf = _first_glob(bundle_dir, "random_forest_model*.pkl")
        xgb_json = _first_glob(bundle_dir, "xgboost_model*.json")
        lstm_pth = _first_glob(bundle_dir, "lstm_model*.pth")
        if not all([fs, ts, rf, xgb_json, lstm_pth]):
            raise FileNotFoundError(
                f"Missing artifacts in {bundle_dir}: "
                f"feature_scaler={fs}, target={ts}, rf={rf}, xgb={xgb_json}, lstm={lstm_pth}"
            )
        feature_scaler = joblib.load(fs)
        target_scaler = joblib.load(ts)
        rf_model = joblib.load(rf)
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(xgb_json))
        input_size = int(getattr(feature_scaler, "n_features_in_", len(FEATURE_COLS)))
        lstm = TaxiLSTM(input_size=input_size)
        try:
            state = torch.load(str(lstm_pth), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(lstm_pth), map_location="cpu")
        lstm.load_state_dict(state)
        lstm.eval()
        return cls(
            bundle_id=bundle_id,
            bundle_dir=bundle_dir,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            xgb_model=xgb_model,
            rf_model=rf_model,
            lstm=lstm,
            input_size=input_size,
        )

    def predict_from_scaled_window(self, scaled_window: np.ndarray) -> Dict[str, float]:
        """
        Feature window shape (TIME_STEPS, n_features), **already scaled** with this
        bundle's `feature_scaler` (see `build_scaled_window`). Outputs are mapped
        through this bundle's `target_scaler.inverse_transform`.
        """
        flat = scaled_window.reshape(1, -1)
        xgb_scaled = self.xgb_model.predict(flat).reshape(-1, 1)
        rf_scaled = self.rf_model.predict(flat).reshape(-1, 1)

        tensor = torch.from_numpy(scaled_window.reshape(1, TIME_STEPS, -1).astype(np.float32))
        with torch.no_grad():
            lstm_scaled = self.lstm(tensor).cpu().numpy()

        def inv(y):
            return float(self.target_scaler.inverse_transform(y.reshape(-1, 1)).ravel()[0])

        vx = inv(xgb_scaled)
        vr = inv(rf_scaled)
        vl = inv(lstm_scaled)
        return {
            "xgboost": vx,
            "random_forest": vr,
            "lstm": vl,
            "ensemble": float(np.mean([vx, vr, vl])),
        }


class NestedModelRegistry:
    """Lazy nested registry: bundle_id -> ModelBundle."""

    def __init__(self, models_root: Path):
        self._models_root = models_root
        self._cache: Dict[str, ModelBundle] = {}
        self._errors: Dict[str, str] = {}

    def list_bundles(self) -> List[str]:
        if not self._models_root.is_dir():
            return []
        names = sorted(p.name for p in self._models_root.iterdir() if p.is_dir())
        return [n for n in names if n not in _EXCLUDED_BUNDLE_IDS]

    def loaded_bundle_ids(self) -> List[str]:
        return list(self._cache.keys())

    def get(self, bundle_id: str) -> Tuple[Optional[ModelBundle], Optional[str]]:
        if bundle_id in self._cache:
            return self._cache[bundle_id], None
        path = self._models_root / bundle_id
        if not path.is_dir():
            msg = f"No model bundle directory: {path}"
            self._errors[bundle_id] = msg
            return None, msg
        try:
            bundle = ModelBundle.load(path)
            self._cache[bundle_id] = bundle
            return bundle, None
        except Exception as e:
            msg = str(e)
            self._errors[bundle_id] = msg
            return None, msg


def apply_feature_scaling(feature_scaler, raw: np.ndarray) -> np.ndarray:
    """Use bundle feature scaler unless REALTIME_FEATURES_RAW=1."""
    if os.environ.get("REALTIME_FEATURES_RAW", "").lower() in ("1", "true", "yes", "on"):
        return raw
    return feature_scaler.transform(raw)


def build_scaled_window(
    df,
    zone_col: str,
    zone_value: int,
    feature_scaler,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Returns the last TIME_STEPS rows of FEATURE_COLS for the zone, transformed
    with this bundle's **feature_scaler** (MinMax on training features).

    Set REALTIME_FEATURES_RAW=1 to skip scaling (only if artifacts match raw
    `create_dataset(data, …)` training from bigdata.ipynb).
    """
    sub = df[df[zone_col] == zone_value].copy()
    if "bucket_time" in sub.columns:
        sub = sub.sort_values("bucket_time", kind="mergesort").reset_index(drop=True)
    if len(sub) < TIME_STEPS:
        return None, f"Need at least {TIME_STEPS} rows for zone {zone_value}, got {len(sub)}"
    tail = sub.tail(TIME_STEPS)
    raw = tail[FEATURE_COLS].values.astype(np.float64)
    return apply_feature_scaling(feature_scaler, raw), None
