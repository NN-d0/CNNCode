import argparse
import csv
import json
from pathlib import Path

import numpy as np


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_split_npz(path: Path, data: dict):
    np.savez_compressed(
        path,
        iq=data["iq"],
        label=data["label"],
        label_name=data["label_name"],
        snr_db=data["snr_db"],
        channel_model=data["channel_model"],
    )


def append_chunk(buffer: dict, iq, label, label_name, snr_db, channel_model):
    buffer["iq"].append(iq)
    buffer["label"].append(label)
    buffer["label_name"].append(label_name)
    buffer["snr_db"].append(snr_db)
    buffer["channel_model"].append(channel_model)


def concat_buffer(buffer: dict):
    if not buffer["iq"]:
        return {
            "iq": np.empty((0, 2, 0), dtype=np.float32),
            "label": np.empty((0,), dtype=np.int64),
            "label_name": np.empty((0,), dtype="<U16"),
            "snr_db": np.empty((0,), dtype=np.int16),
            "channel_model": np.empty((0,), dtype="<U32"),
        }

    return {
        "iq": np.concatenate(buffer["iq"], axis=0).astype(np.float32),
        "label": np.concatenate(buffer["label"], axis=0).astype(np.int64),
        "label_name": np.concatenate(buffer["label_name"], axis=0).astype("<U16"),
        "snr_db": np.concatenate(buffer["snr_db"], axis=0).astype(np.int16),
        "channel_model": np.concatenate(buffer["channel_model"], axis=0).astype("<U32"),
    }


def shuffle_split(data: dict, seed: int):
    size = len(data["label"])
    if size <= 1:
        return data

    rng = np.random.default_rng(seed)
    idx = np.arange(size)
    rng.shuffle(idx)

    return {
        "iq": data["iq"][idx],
        "label": data["label"][idx],
        "label_name": data["label_name"][idx],
        "snr_db": data["snr_db"][idx],
        "channel_model": data["channel_model"][idx],
    }


def summarize_by_label_and_snr(data: dict):
    summary = {}
    labels = data["label_name"]
    snrs = data["snr_db"]

    for label_name, snr_db in zip(labels, snrs):
        key = (str(label_name), int(snr_db))
        summary[key] = summary.get(key, 0) + 1

    return summary


def write_split_summary_csv(path: Path, train_data: dict, val_data: dict, test_data: dict):
    train_summary = summarize_by_label_and_snr(train_data)
    val_summary = summarize_by_label_and_snr(val_data)
    test_summary = summarize_by_label_and_snr(test_data)

    all_keys = sorted(set(train_summary.keys()) | set(val_summary.keys()) | set(test_summary.keys()))

    fieldnames = ["label_name", "snr_db", "train_count", "val_count", "test_count", "total_count"]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for label_name, snr_db in all_keys:
            train_count = train_summary.get((label_name, snr_db), 0)
            val_count = val_summary.get((label_name, snr_db), 0)
            test_count = test_summary.get((label_name, snr_db), 0)

            writer.writerow({
                "label_name": label_name,
                "snr_db": snr_db,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "total_count": train_count + val_count + test_count,
            })


def build_arg_parser():
    parser = argparse.ArgumentParser(description="将原始 I/Q 数据集切分为 train/val/test")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="ai-research 根目录，默认自动推断"
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--seed", type=int, default=20260325, help="随机种子")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test 比例之和必须等于 1.0")

    project_root = Path(args.project_root).resolve()
    dataset_root = project_root / "dataset"

    raw_dir = dataset_root / "raw"
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"
    metadata_dir = dataset_root / "metadata"

    ensure_dir(train_dir)
    ensure_dir(val_dir)
    ensure_dir(test_dir)
    ensure_dir(metadata_dir)

    raw_files = sorted(raw_dir.glob("*.npz"))
    if not raw_files:
        raise FileNotFoundError(f"未找到原始数据文件，请先生成数据集：{raw_dir}")

    train_buffer = {"iq": [], "label": [], "label_name": [], "snr_db": [], "channel_model": []}
    val_buffer = {"iq": [], "label": [], "label_name": [], "snr_db": [], "channel_model": []}
    test_buffer = {"iq": [], "label": [], "label_name": [], "snr_db": [], "channel_model": []}

    total_raw_samples = 0

    for file_idx, file_path in enumerate(raw_files):
        data = np.load(file_path, allow_pickle=False)

        iq = data["iq"]
        label = data["label"]
        label_name = data["label_name"].astype("<U16")
        snr_db = data["snr_db"]
        channel_model = data["channel_model"].astype("<U32")

        n = len(label)
        total_raw_samples += n

        rng = np.random.default_rng(args.seed + file_idx)
        idx = np.arange(n)
        rng.shuffle(idx)

        train_end = int(n * args.train_ratio)
        val_end = train_end + int(n * args.val_ratio)

        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]

        append_chunk(
            train_buffer,
            iq[train_idx],
            label[train_idx],
            label_name[train_idx],
            snr_db[train_idx],
            channel_model[train_idx],
        )

        append_chunk(
            val_buffer,
            iq[val_idx],
            label[val_idx],
            label_name[val_idx],
            snr_db[val_idx],
            channel_model[val_idx],
        )

        append_chunk(
            test_buffer,
            iq[test_idx],
            label[test_idx],
            label_name[test_idx],
            snr_db[test_idx],
            channel_model[test_idx],
        )

        print(
            f"[OK] 已切分 {file_path.name} -> "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    train_data = concat_buffer(train_buffer)
    val_data = concat_buffer(val_buffer)
    test_data = concat_buffer(test_buffer)

    train_data = shuffle_split(train_data, args.seed + 100)
    val_data = shuffle_split(val_data, args.seed + 200)
    test_data = shuffle_split(test_data, args.seed + 300)

    save_split_npz(train_dir / "train.npz", train_data)
    save_split_npz(val_dir / "val.npz", val_data)
    save_split_npz(test_dir / "test.npz", test_data)

    split_summary = {
        "raw_file_count": len(raw_files),
        "raw_total_samples": total_raw_samples,
        "train_count": int(len(train_data["label"])),
        "val_count": int(len(val_data["label"])),
        "test_count": int(len(test_data["label"])),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }

    write_json(metadata_dir / "split_summary.json", split_summary)
    write_split_summary_csv(
        metadata_dir / "split_summary.csv",
        train_data,
        val_data,
        test_data
    )

    print("数据集切分完成")
    print(f"project_root: {project_root}")
    print(f"train: {train_dir / 'train.npz'}")
    print(f"val:   {val_dir / 'val.npz'}")
    print(f"test:  {test_dir / 'test.npz'}")
    print(f"train_count: {len(train_data['label'])}")
    print(f"val_count: {len(val_data['label'])}")
    print(f"test_count: {len(test_data['label'])}")

if __name__ == "__main__":
    main()