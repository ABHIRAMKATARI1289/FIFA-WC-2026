import pandas as pd
import numpy as np
import pickle
import os
import json
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.metrics import ClassificationQualityMetric

PROCESSED = "data/processed"
MODELS    = "models"
REPORTS   = "reports"
os.makedirs(REPORTS, exist_ok=True)

FEATURE_COLS = [
    "home_win_rate", "home_goals_scored_avg", "home_goals_conceded_avg",
    "away_win_rate", "away_goals_scored_avg", "away_goals_conceded_avg",
    "h2h_home_win_rate", "h2h_matches",
    "goal_diff_avg", "win_rate_diff", "is_knockout"
]

def encode_outcome(df):
    mapping = {"home_win": 2, "draw": 1, "away_win": 0}
    return df["outcome"].map(mapping)

def generate_drift_report():
    """Compare feature distributions between WC eras — detect drift across seasons."""
    print("Loading features...")
    df = pd.read_csv(f"{PROCESSED}/features.csv", parse_dates=["date"])
    df = df.dropna(subset=FEATURE_COLS)

    # Reference: 1990-2006, Current: 2010-2014
    reference = df[df["year"] <= 2006][FEATURE_COLS].reset_index(drop=True)
    current   = df[df["year"] >= 2010][FEATURE_COLS].reset_index(drop=True)

    print(f"  Reference period (1990-2006): {len(reference)} matches")
    print(f"  Current period (2010-2014):   {len(current)} matches")

    report = Report(metrics=[DataDriftTable(), DatasetDriftMetric()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(f"{REPORTS}/drift_report.html")
    print(f"  Saved drift report → reports/drift_report.html ✅")

    # Extract drift summary
    result = report.as_dict()
    drifted = []
    for metric in result.get("metrics", []):
        if "result" in metric:
            r = metric["result"]
            if isinstance(r, dict) and r.get("dataset_drift"):
                drifted.append("Dataset drift detected")
            if "drift_by_columns" in r:
                for col, info in r["drift_by_columns"].items():
                    if info.get("drift_detected"):
                        drifted.append(f"  ⚠ Drift in: {col}")

    if drifted:
        print("\nDrift Summary:")
        for d in drifted:
            print(f"  {d}")
    else:
        print("\n✅ No significant drift detected between eras")

    return result

def generate_classification_report():
    """Generate model performance report on test set (WC 2014)."""
    print("\nGenerating classification report...")

    df = pd.read_csv(f"{PROCESSED}/features.csv", parse_dates=["date"])
    df = df.dropna(subset=FEATURE_COLS)

    with open(f"{MODELS}/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    test = df[df["year"] == 2014].copy()
    X_test = test[FEATURE_COLS]
    y_test = encode_outcome(test)

    probs  = model.predict_proba(X_test)
    preds  = model.predict(X_test)

    class_map = {c: i for i, c in enumerate(model.classes_)}

    test_eval = test[["home_team", "away_team", "outcome", "year"]].copy()
    test_eval["target"]     = y_test.values
    test_eval["prediction"] = preds

    # Add probabilities for each class
    for c in model.classes_:
        test_eval[f"prob_{c}"] = probs[:, class_map[c]]

    report = Report(metrics=[ClassificationQualityMetric()])
    report.run(
        reference_data=None,
        current_data=test_eval[["target", "prediction"]]
    )
    report.save_html(f"{REPORTS}/classification_report.html")
    print(f"  Saved classification report → reports/classification_report.html ✅")

    # Print quick summary
    correct = (test_eval["target"] == test_eval["prediction"]).sum()
    total   = len(test_eval)
    print(f"  Accuracy on WC 2014: {correct}/{total} = {correct/total:.1%}")

    return test_eval

def save_monitoring_summary(drift_result, test_eval):
    """Save a JSON summary for the dashboard to consume."""
    correct = (test_eval["target"] == test_eval["prediction"]).sum()
    total   = len(test_eval)

    summary = {
        "model_accuracy_2014": round(correct / total, 4),
        "total_test_matches":  total,
        "correct_predictions": int(correct),
        "reports": {
            "drift":          "reports/drift_report.html",
            "classification": "reports/classification_report.html"
        }
    }

    with open(f"{REPORTS}/monitoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMonitoring summary saved → reports/monitoring_summary.json ✅")
    print(f"\n── Summary ──────────────────────────────────────")
    print(f"  Model accuracy on WC 2014: {correct/total:.1%}")
    print(f"  Correct predictions: {correct}/{total}")
    print(f"  Drift report: reports/drift_report.html")
    print(f"  Classification report: reports/classification_report.html")

if __name__ == "__main__":
    drift_result = generate_drift_report()
    test_eval    = generate_classification_report()
    save_monitoring_summary(drift_result, test_eval)
    print("\nMonitoring complete. Open HTML reports in browser to explore.")