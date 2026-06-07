import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
import xgboost as xgb

PROCESSED = "data/processed"
MODELS = "models"
os.makedirs(MODELS, exist_ok=True)

FEATURE_COLS = [
    "home_win_rate", "home_goals_scored_avg", "home_goals_conceded_avg",
    "away_win_rate", "away_goals_scored_avg", "away_goals_conceded_avg",
    "h2h_home_win_rate", "h2h_matches",
    "goal_diff_avg", "win_rate_diff", "is_knockout"
]

def encode_outcome(df):
    mapping = {"home_win": 2, "draw": 1, "away_win": 0}
    return df["outcome"].map(mapping)

def evaluate(model, X_test, y_test, label="Model"):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    brier = np.mean([
        brier_score_loss((y_test == c).astype(int), probs[:, i])
        for i, c in enumerate(model.classes_)
    ])
    print(f"\n{label}")
    print(f"  Accuracy:    {acc:.3f}")
    print(f"  Brier Score: {brier:.4f}  (lower = better, 0.25 = random)")
    return acc, brier

if __name__ == "__main__":
    print("Loading features...")
    df = pd.read_csv(f"{PROCESSED}/features.csv", parse_dates=["date"])
    df = df.dropna(subset=FEATURE_COLS)
    print(f"  Total matches: {len(df)}")

    # Time-based split — never random split on time series data
    train = df[df["year"] <= 2010].copy()
    test  = df[df["year"] == 2014].copy()
    print(f"  Train: {len(train)} matches (1990–2010)")
    print(f"  Test:  {len(test)} matches (2014)")

    X_train = train[FEATURE_COLS]
    y_train = encode_outcome(train)
    X_test  = test[FEATURE_COLS]
    y_test  = encode_outcome(test)

    print(f"\nClass distribution (train):")
    print(f"  {y_train.value_counts().to_dict()}  (2=home_win, 1=draw, 0=away_win)")

    # ── Baseline: Logistic Regression ─────────────────────────────────────────
    print("\nTraining Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc, lr_brier = evaluate(lr, X_test, y_test, "Logistic Regression")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_acc, xgb_brier = evaluate(xgb_model, X_test, y_test, "XGBoost")

    # ── Feature importance ────────────────────────────────────────────────────
    print("\nXGBoost Feature Importance:")
    importance = pd.Series(
        xgb_model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    for feat, score in importance.items():
        print(f"  {feat:<30} {score:.4f}")

    # ── Save best model ───────────────────────────────────────────────────────
    best_model = xgb_model if xgb_brier < lr_brier else lr
    best_name  = "XGBoost" if xgb_brier < lr_brier else "LogisticRegression"
    print(f"\nBest model: {best_name} (Brier: {min(xgb_brier, lr_brier):.4f})")

    with open(f"{MODELS}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(f"{MODELS}/feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    print(f"Model saved to models/best_model.pkl ✅")

    # ── Sample prediction ─────────────────────────────────────────────────────
    print("\n── Sample prediction: Brazil vs Germany (2014 final context) ──")
    sample = pd.DataFrame([{
        "home_win_rate": 0.65,
        "home_goals_scored_avg": 2.1,
        "home_goals_conceded_avg": 0.8,
        "away_win_rate": 0.70,
        "away_goals_scored_avg": 2.3,
        "away_goals_conceded_avg": 0.6,
        "h2h_home_win_rate": 0.4,
        "h2h_matches": 5,
        "goal_diff_avg": -0.4,
        "win_rate_diff": -0.05,
        "is_knockout": 1
    }])
    probs = best_model.predict_proba(sample)[0]
    classes = best_model.classes_
    label_map = {2: "Home Win", 1: "Draw", 0: "Away Win"}
    for c, p in zip(classes, probs):
        print(f"  {label_map[c]:<12} {p:.1%}")