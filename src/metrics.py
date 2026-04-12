import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)


def evaluate(df, threshold=None, optimize_for="f1"):

    y_true = df["is_anomaly"].astype(int).values
    y_scores = df["score"].values

    # ROC hesapla
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    best_threshold = None

    # ---------------------------------
    # THRESHOLD OPTIMIZATION (Validation Mode)
    # ---------------------------------
    if threshold is None:

        if optimize_for == "f1":
            f1_scores = []

            for t in thresholds:
                y_pred = (y_scores > t).astype(int)
                f1_scores.append(f1_score(y_true, y_pred))

            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]

        elif optimize_for == "youden":
            youden_scores = tpr - fpr
            best_idx = np.argmax(youden_scores)
            best_threshold = thresholds[best_idx]

        else:
            raise ValueError("optimize_for must be 'f1' or 'youden'")

        used_threshold = best_threshold

    # ---------------------------------
    # FIXED THRESHOLD (Test Mode)
    # ---------------------------------
    else:
        used_threshold = threshold

    # Final predictions
    y_pred = (y_scores > used_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "best_threshold": best_threshold,
        "used_threshold": used_threshold,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        },
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr
        }
    }
