from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBoostOutcomeModel:
    """Small supervised wrapper reserved for larger historical datasets."""

    def __init__(self) -> None:
        self.model = XGBClassifier(eval_metric="mlogloss", n_estimators=50, max_depth=3)
        self.encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        encoded = self.encoder.fit_transform(y)
        self.model.fit(x, encoded)
        self.is_fitted = True

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("XGBoostOutcomeModel must be fitted before prediction.")
        probabilities = self.model.predict_proba(x)
        return pd.DataFrame(probabilities, columns=self.encoder.classes_)

