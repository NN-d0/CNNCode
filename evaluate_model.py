import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LABEL_MAP = {
    "AM": 0,
    "FM": 1,
    "BPSK": 2,
    "QPSK": 3,
    "16QAM": 4,
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_label_map(label_map_path: Path) -> Dict[str, int]:
    if label_map_path.exists():
        with label_map_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_LABEL_MAP


def invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in label_map.items()}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="评估 1D-CNN 测试结果并输出图表与分类报告")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="ai-research 根目录，默认自动推断",
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        default="",
        help="测试集预测结果文件，默认使用 reports/test_predictions.npz",
    )
    parser.add_argument(
        "--label-map-file",
        type=str,
        default="",
        help="标签映射文件，默认使用 dataset/metadata/label_map.json",
    )
    return parser


def load_predictions(predictions_path: Path) -> Dict[str, np.ndarray]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"未找到预测结果文件：{predictions_path}")

    data = np.load(predictions_path, allow_pickle=False)

    required_keys = ["pred_label", "true_label", "snr_db"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"{predictions_path} 中缺少必需字段：{key}")

    result = {
        "pred_label": data["pred_label"].astype(np.int64),
        "true_label": data["true_label"].astype(np.int64),
        "snr_db": data["snr_db"].astype(np.int16),
    }

    if "true_label_name" in data:
        result["true_label_name"] = data["true_label_name"].astype(str)

    if "channel_model" in data:
        result["channel_model"] = data["channel_model"].astype(str)

    if "probabilities" in data:
        result["probabilities"] = data["probabilities"].astype(np.float32)

    return result


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_classification_report(
    cm: np.ndarray,
    idx_to_name: Dict[int, str],
) -> Tuple[List[Dict], Dict]:
    """
    返回：
    1. 每类指标列表
    2. 总体指标字典
    """
    num_classes = cm.shape[0]
    per_class_rows = []

    total_samples = int(cm.sum())
    correct = int(np.trace(cm))
    overall_accuracy = safe_div(correct, total_samples)

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for cls_idx in range(num_classes):
        tp = int(cm[cls_idx, cls_idx])
        fp = int(cm[:, cls_idx].sum() - tp)
        fn = int(cm[cls_idx, :].sum() - tp)
        support = int(cm[cls_idx, :].sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        per_class_rows.append({
            "label_id": cls_idx,
            "label_name": idx_to_name.get(cls_idx, f"class_{cls_idx}"),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
            "support": support,
        })

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    support_sum = sum(supports) if supports else 0
    weighted_precision = (
        float(np.sum(np.array(precisions) * np.array(supports)) / support_sum) if support_sum > 0 else 0.0
    )
    weighted_recall = (
        float(np.sum(np.array(recalls) * np.array(supports)) / support_sum) if support_sum > 0 else 0.0
    )
    weighted_f1 = (
        float(np.sum(np.array(f1s) * np.array(supports)) / support_sum) if support_sum > 0 else 0.0
    )

    summary = {
        "total_samples": total_samples,
        "correct_samples": correct,
        "accuracy": round(overall_accuracy, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_precision": round(weighted_precision, 6),
        "weighted_recall": round(weighted_recall, 6),
        "weighted_f1": round(weighted_f1, 6),
    }

    return per_class_rows, summary


def write_classification_report_csv(
    output_path: Path,
    per_class_rows: List[Dict],
    summary: Dict,
):
    fieldnames = ["label_id", "label_name", "precision", "recall", "f1_score", "support"]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_class_rows:
            writer.writerow(row)

        writer.writerow({})
        writer.writerow({"label_name": "accuracy", "precision": summary["accuracy"]})
        writer.writerow({"label_name": "macro_precision", "precision": summary["macro_precision"]})
        writer.writerow({"label_name": "macro_recall", "precision": summary["macro_recall"]})
        writer.writerow({"label_name": "macro_f1", "precision": summary["macro_f1"]})
        writer.writerow({"label_name": "weighted_precision", "precision": summary["weighted_precision"]})
        writer.writerow({"label_name": "weighted_recall", "precision": summary["weighted_recall"]})
        writer.writerow({"label_name": "weighted_f1", "precision": summary["weighted_f1"]})


def write_classification_report_json(
    output_path: Path,
    per_class_rows: List[Dict],
    summary: Dict,
):
    data = {
        "per_class": per_class_rows,
        "summary": summary,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_confusion_matrix(
    cm: np.ndarray,
    idx_to_name: Dict[int, str],
    output_path: Path,
):
    labels = [idx_to_name.get(i, f"class_{i}") for i in range(cm.shape[0])]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            plt.text(
                j,
                i,
                str(value),
                horizontalalignment="center",
                color="white" if value > threshold else "black",
                fontsize=10,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_snr_accuracy(y_true: np.ndarray, y_pred: np.ndarray, snr_db: np.ndarray) -> List[Dict]:
    result = []
    unique_snrs = sorted(int(x) for x in np.unique(snr_db))

    for snr in unique_snrs:
        mask = snr_db == snr
        count = int(mask.sum())
        if count == 0:
            continue

        correct = int((y_true[mask] == y_pred[mask]).sum())
        accuracy = safe_div(correct, count)

        result.append({
            "snr_db": snr,
            "sample_count": count,
            "correct_count": correct,
            "accuracy": round(float(accuracy), 6),
        })

    return result


def write_snr_accuracy_csv(output_path: Path, rows: List[Dict]):
    fieldnames = ["snr_db", "sample_count", "correct_count", "accuracy"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_snr_accuracy_curve(rows: List[Dict], output_path: Path):
    if not rows:
        return

    x = [row["snr_db"] for row in rows]
    y = [row["accuracy"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy under Different SNR")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_per_class_metrics(
    per_class_rows: List[Dict],
    output_path: Path,
):
    labels = [row["label_name"] for row in per_class_rows]
    precision = [row["precision"] for row in per_class_rows]
    recall = [row["recall"] for row in per_class_rows]
    f1_score = [row["f1_score"] for row in per_class_rows]

    x = np.arange(len(labels))
    width = 0.24

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, precision, width=width, label="Precision")
    plt.bar(x, recall, width=width, label="Recall")
    plt.bar(x + width, f1_score, width=width, label="F1-score")
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-class Precision / Recall / F1")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_evaluation_summary(
    predictions_path: Path,
    label_map_path: Path,
    report_summary: Dict,
    snr_rows: List[Dict],
) -> Dict:
    return {
        "predictions_file": str(predictions_path),
        "label_map_file": str(label_map_path),
        "accuracy": report_summary["accuracy"],
        "macro_precision": report_summary["macro_precision"],
        "macro_recall": report_summary["macro_recall"],
        "macro_f1": report_summary["macro_f1"],
        "weighted_precision": report_summary["weighted_precision"],
        "weighted_recall": report_summary["weighted_recall"],
        "weighted_f1": report_summary["weighted_f1"],
        "total_samples": report_summary["total_samples"],
        "snr_min": min((row["snr_db"] for row in snr_rows), default=None),
        "snr_max": max((row["snr_db"] for row in snr_rows), default=None),
        "snr_points": len(snr_rows),
    }


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    report_dir = project_root / "reports"
    metadata_dir = project_root / "dataset" / "metadata"

    ensure_dir(report_dir)
    ensure_dir(metadata_dir)

    predictions_path = (
        Path(args.predictions_file).resolve()
        if args.predictions_file
        else (report_dir / "test_predictions.npz")
    )
    label_map_path = (
        Path(args.label_map_file).resolve()
        if args.label_map_file
        else (metadata_dir / "label_map.json")
    )

    label_map = load_label_map(label_map_path)
    idx_to_name = invert_label_map(label_map)

    predictions = load_predictions(predictions_path)
    y_pred = predictions["pred_label"]
    y_true = predictions["true_label"]
    snr_db = predictions["snr_db"]

    num_classes = max(int(max(label_map.values())) + 1, int(y_true.max()) + 1, int(y_pred.max()) + 1)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    per_class_rows, summary = compute_classification_report(cm, idx_to_name)
    snr_rows = compute_snr_accuracy(y_true, y_pred, snr_db)

    classification_csv_path = report_dir / "classification_report.csv"
    classification_json_path = report_dir / "classification_report.json"
    confusion_matrix_path = report_dir / "confusion_matrix.png"
    snr_curve_path = report_dir / "snr_accuracy_curve.png"
    snr_csv_path = report_dir / "snr_accuracy_report.csv"
    per_class_metrics_path = report_dir / "per_class_metrics.png"
    evaluation_summary_path = report_dir / "evaluation_summary.json"
    confusion_matrix_npy_path = report_dir / "confusion_matrix.npy"

    write_classification_report_csv(classification_csv_path, per_class_rows, summary)
    write_classification_report_json(classification_json_path, per_class_rows, summary)
    plot_confusion_matrix(cm, idx_to_name, confusion_matrix_path)
    plot_snr_accuracy_curve(snr_rows, snr_curve_path)
    write_snr_accuracy_csv(snr_csv_path, snr_rows)
    plot_per_class_metrics(per_class_rows, per_class_metrics_path)

    np.save(confusion_matrix_npy_path, cm)

    evaluation_summary = build_evaluation_summary(
        predictions_path=predictions_path,
        label_map_path=label_map_path,
        report_summary=summary,
        snr_rows=snr_rows,
    )

    with evaluation_summary_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)

    print("模型评估完成")
    print(f"predictions_file: {predictions_path}")
    print(f"label_map_file:   {label_map_path}")
    print(f"accuracy:         {summary['accuracy']:.4f}")
    print(f"macro_f1:         {summary['macro_f1']:.4f}")
    print(f"classification_report.csv: {classification_csv_path}")
    print(f"classification_report.json:{classification_json_path}")
    print(f"confusion_matrix.png:      {confusion_matrix_path}")
    print(f"snr_accuracy_curve.png:    {snr_curve_path}")
    print(f"snr_accuracy_report.csv:   {snr_csv_path}")
    print(f"per_class_metrics.png:     {per_class_metrics_path}")
    print(f"evaluation_summary.json:   {evaluation_summary_path}")

if __name__ == "__main__":
    main()