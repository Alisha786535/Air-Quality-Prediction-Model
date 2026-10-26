"""Train + evaluate + save. Run from project root:  python -m src.train
Uses data/raw/*.csv if present, otherwise falls back to synthetic data (flagged in the bundle)."""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.base import clone

from ...src.data import load_raw, make_synthetic
from ...src.features import CATS, ALERT_LEVEL, HORIZONS, build_features, feature_columns
from ...src.forecast import make_forecast, risk_ranking
from ...src.models import classifiers, regressors

ROOT = Path(__file__).resolve().parent.parent
LABELS = list(range(len(CATS)))


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main():
    daily = load_raw(ROOT / "data/raw")
    source = "real"
    if daily is None or daily.empty:
        print("!! No CSVs in data/raw -> using SYNTHETIC data (pipeline test only)")
        daily, source = make_synthetic(), "synthetic"
    daily.to_csv(ROOT / "data/processed/daily.csv", index=False)

    base_cols = [c for c in ["aqi", "pm25", "pm10", "temp", "humidity", "wind_speed", "pressure"] if c in daily]
    cities = sorted(daily["city"].unique())
    feats = build_features(daily, base_cols, cities)
    fcols = feature_columns(feats)
    print(f"{source} data | {len(cities)} cities | {len(daily)} city-days | {len(fcols)} features")

    # chronological split 70/15/15 by calendar date (no shuffling -> no leakage)
    dates = np.sort(feats["date"].unique())
    d_tr, d_va = dates[int(len(dates) * .70)], dates[int(len(dates) * .85)]

    bundle = dict(feature_cols=fcols, base_cols=base_cols, cities=cities, clf={}, reg={},
                  data_source=source, metrics={}, selection={}, split=dict(val_start=str(d_tr)[:10], test_start=str(d_va)[:10]))
    all_metrics = []
    for h in HORIZONS:
        tc, ta = f"target_cat_t{h}", f"target_aqi_t{h}"
        sub = feats.dropna(subset=fcols + [tc, ta])
        tr, va, te = (sub[sub.date < d_tr], sub[(sub.date >= d_tr) & (sub.date < d_va)], sub[sub.date >= d_va])
        ytr, yva = tr[tc].astype(int), va[tc].astype(int)

        # ---- model selection on VALIDATION (macro-F1 / RMSE), then refit on train+val
        best_c = max(classifiers().items(), key=lambda kv: f1_score(
            yva, kv[1].fit(tr[fcols], ytr).predict(va[fcols]), average="macro", labels=LABELS, zero_division=0))[0]
        best_r = min(regressors().items(), key=lambda kv: rmse(
            va[ta], kv[1].fit(tr[fcols], tr[ta]).predict(va[fcols])))[0]
        trv = pd.concat([tr, va])
        clf = classifiers()[best_c]
        clf.fit(trv[fcols], trv[tc].astype(int))
        reg = regressors()[best_r].fit(trv[fcols], trv[ta])
        bundle["clf"][h], bundle["reg"][h] = clf, reg
        bundle["selection"][h] = dict(classifier=best_c, regressor=best_r)

        # ---- test metrics vs persistence baseline ("tomorrow = today")
        yte, pte = te[tc].astype(int), clf.predict(te[fcols])
        from ...src.features import aqi_to_cat
        base_cat = aqi_to_cat(te["aqi_now"].values).astype(int)
        pred_aqi = reg.predict(te[fcols])
        ya, pa = (yte >= ALERT_LEVEL), (pte >= ALERT_LEVEL)
        m = dict(horizon_days=h, n_test=len(te), classifier=best_c, regressor=best_r,
                 accuracy=accuracy_score(yte, pte), macro_f1=f1_score(yte, pte, average="macro", labels=LABELS, zero_division=0),
                 weighted_f1=f1_score(yte, pte, average="weighted", zero_division=0),
                 alert_precision=precision_score(ya, pa, zero_division=0), alert_recall=recall_score(ya, pa, zero_division=0),
                 aqi_rmse=rmse(te[ta], pred_aqi), aqi_mae=float(mean_absolute_error(te[ta], pred_aqi)),
                 baseline_accuracy=accuracy_score(yte, base_cat),
                 baseline_macro_f1=f1_score(yte, base_cat, average="macro", labels=LABELS, zero_division=0),
                 baseline_aqi_rmse=rmse(te[ta], te["aqi_now"]), baseline_aqi_mae=float(mean_absolute_error(te[ta], te["aqi_now"])))
        all_metrics.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
        print(f"t+{h}: {best_c:>20} acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} | RMSE={m['aqi_rmse']:.1f} MAE={m['aqi_mae']:.1f} "
              f"| persistence acc={m['baseline_accuracy']:.3f} RMSE={m['baseline_aqi_rmse']:.1f}")

        if h == 1:  # explainability: permutation importance of the 1-day-ahead classifier on the test set
            s = te.sample(min(600, len(te)), random_state=0)
            pi = permutation_importance(clf, s[fcols], s[tc].astype(int), n_repeats=2, random_state=0,
                                        scoring="f1_macro", n_jobs=1)
            bundle["importance"] = (pd.Series(pi.importances_mean, index=fcols).sort_values(ascending=False).head(15).round(4).to_dict())

    bundle["metrics"] = all_metrics
    joblib.dump(bundle, ROOT / "models/aqi_model.joblib", compress=3)
    (ROOT / "reports/metrics.json").write_text(json.dumps(dict(data_source=source, split=bundle["split"],
                                               metrics=all_metrics, top_features=bundle["importance"]), indent=2))

    fc = make_forecast(bundle, daily)
    fc.drop(columns="category_index").to_csv(ROOT / "outputs/forecast.csv", index=False)
    risk_ranking(fc).to_csv(ROOT / "outputs/city_risk_ranking.csv", index=False)
    print("\nSaved: models/aqi_model.joblib, reports/metrics.json, outputs/forecast.csv, outputs/city_risk_ranking.csv")
    print(risk_ranking(fc).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
