import argparse
import json

import numpy as np


def icc_3_1(reference, prediction):
    ratings = np.column_stack((reference, prediction)).astype(float)
    n, k = ratings.shape
    row_means = ratings.mean(axis=1)
    column_means = ratings.mean(axis=0)
    grand_mean = ratings.mean()
    ms_rows = k * np.square(row_means - grand_mean).sum() / (n - 1)
    residual = ratings - row_means[:, None] - column_means[None, :] + grand_mean
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    return (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)


def metrics(reference, prediction):
    reference = np.asarray(reference, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    recalls, f1_scores = [], []
    for score in range(1, 5):
        true_positive = np.sum((reference == score) & (prediction == score))
        false_negative = np.sum((reference == score) & (prediction != score))
        false_positive = np.sum((reference != score) & (prediction == score))
        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        recalls.append(recall)
        f1_scores.append(2 * precision * recall / max(precision + recall, np.finfo(float).eps))
    return {
        "mae": float(np.abs(reference - prediction).mean()),
        "top1": float(np.mean(reference == prediction)),
        "pearson_r": float(np.corrcoef(reference, prediction)[0, 1]),
        "icc_3_1": float(icc_3_1(reference, prediction)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "recall_by_score": [float(value) for value in recalls],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute the DV-FraS manuscript metrics.")
    parser.add_argument("predictions", help="Prediction JSON written by train_dvfras.py.")
    args = parser.parse_args()
    with open(args.predictions, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    reference_exams = np.asarray([row["target"] for row in rows])
    prediction_exams = np.asarray([row["prediction"] for row in rows])
    result = metrics(reference_exams.reshape(-1), prediction_exams.reshape(-1))
    reference_total = reference_exams.sum(axis=(1, 2))
    prediction_total = prediction_exams.sum(axis=(1, 2))
    result["total_score_exact_accuracy"] = float(np.mean(reference_total == prediction_total))
    result["total_score_accuracy_within_1"] = float(
        np.mean(np.abs(reference_total - prediction_total) <= 1)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
