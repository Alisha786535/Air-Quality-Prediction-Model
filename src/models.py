"""Model zoo. Wrappers live here (not in train.py) so joblib can unpickle them from the Streamlit app."""
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # xgboost optional
    XGBClassifier = XGBRegressor = None


class XGBClf:
    """XGBoost needs contiguous labels 0..k-1; some AQI classes may be absent in a split, so re-encode."""
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.m_ = XGBClassifier(**self.kw).fit(X, self.le_.transform(y))
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.m_.predict(X))

    def predict_proba(self, X):
        return self.m_.predict_proba(X)


def classifiers():
    m = {
        "RandomForest": RandomForestClassifier(n_estimators=120, min_samples_leaf=3, max_features="sqrt", n_jobs=-1,
                                               class_weight="balanced_subsample", random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                               class_weight="balanced", random_state=42),
    }
    if XGBClassifier is not None:
        m["XGBoost"] = XGBClf(n_estimators=150, learning_rate=0.08, max_depth=5, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=-1, random_state=42)
    return m


def regressors():
    m = {
        "RandomForest": RandomForestRegressor(n_estimators=80, min_samples_leaf=5, max_features=0.4, n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=42),
    }
    if XGBRegressor is not None:
        m["XGBoost"] = XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, subsample=0.8,
                                    colsample_bytree=0.8, n_jobs=-1, random_state=42)
    return m


def prob_alert(model, X, alert_level=3):
    """P(category >= alert_level) from predict_proba, robust to missing classes."""
    p = model.predict_proba(X)
    mask = np.asarray(model.classes_) >= alert_level
    return p[:, mask].sum(axis=1)
