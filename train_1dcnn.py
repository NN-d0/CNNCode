import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


LABEL_MAP = {
    "AM": 0,
    "FM": 1,
    "BPSK": 2,
    "QPSK": 3,
    "16QAM": 4,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_label_map(metadata_dir: Path) -> Dict[str, int]:
    label_map_path = metadata_dir / "label_map.json"
    if label_map_path.exists():
        with label_map_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return LABEL_MAP


def inverse_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in label_map.items()}


class IQDataset(Dataset):
    """
    读取 train.npz / val.npz / test.npz
    支持字段：
    - iq: [N, 2, L]
    - label: [N]
    - label_name: [N]
    - snr_db: [N]
    - channel_model: [N]
    """

    def __init__(self, file_path: Path, normalize: bool = True):
        super().__init__()
        data = np.load(file_path, allow_pickle=False)

        self.iq = data["iq"].astype(np.float32)
        self.label = data["label"].astype(np.int64)
        self.label_name = data["label_name"].astype(str)
        self.snr_db = data["snr_db"].astype(np.int16)
        self.channel_model = data["channel_model"].astype(str)
        self.normalize = normalize

        if len(self.iq.shape) != 3 or self.iq.shape[1] != 2:
            raise ValueError(f"{file_path} 中 iq 形状非法，应为 [N, 2, L]，实际为 {self.iq.shape}")

    def __len__(self):
        return len(self.label)

    def _normalize_sample(self, x: np.ndarray) -> np.ndarray:
        """
        对单条样本做标准化：
        每个通道分别减均值、除标准差
        """
        x = x.copy()
        for c in range(x.shape[0]):
            mean = x[c].mean()
            std = x[c].std() + 1e-6
            x[c] = (x[c] - mean) / std
        return x.astype(np.float32)

    def __getitem__(self, idx: int):
        x = self.iq[idx]
        if self.normalize:
            x = self._normalize_sample(x)

        y = self.label[idx]

        meta = {
            "label_name": self.label_name[idx],
            "snr_db": int(self.snr_db[idx]),
            "channel_model": self.channel_model[idx],
        }

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), meta


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: bool = True):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class OneDCNNClassifier(nn.Module):
    """
    最小可交 1D-CNN
    输入: [B, 2, L]
    输出: [B, num_classes]
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(2, 32, kernel_size=7, pool=True),
            ConvBlock(32, 64, kernel_size=5, pool=True),
            ConvBlock(64, 128, kernel_size=3, pool=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_arg_parser():
    parser = argparse.ArgumentParser(description="训练 1D-CNN 调制识别模型")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="ai-research 根目录，默认自动推断"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 线程数")
    parser.add_argument("--patience", type=int, default=5, help="Early Stop 耐心轮数")
    parser.add_argument("--seed", type=int, default=20260325, help="随机种子")
    parser.add_argument("--device", type=str, default="", help="指定设备")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    correct = (pred == target).sum().item()
    total = target.size(0)
    return correct / max(total, 1)


def build_dataloaders(project_root: Path, batch_size: int, num_workers: int):
    dataset_root = project_root / "dataset"
    train_path = dataset_root / "train" / "train.npz"
    val_path = dataset_root / "val" / "val.npz"
    test_path = dataset_root / "test" / "test.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"未找到训练集文件：{train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"未找到验证集文件：{val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"未找到测试集文件：{test_path}")

    train_dataset = IQDataset(train_path, normalize=True)
    val_dataset = IQDataset(val_path, normalize=True)
    test_dataset = IQDataset(test_path, normalize=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_x, batch_y, _ in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == batch_y).sum().item()
        total_count += batch_size

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_x, batch_y, _ in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == batch_y).sum().item()
        total_count += batch_size

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> Dict[str, np.ndarray]:
    model.eval()

    probs_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []
    label_name_list: List[np.ndarray] = []
    snr_list: List[np.ndarray] = []
    channel_list: List[np.ndarray] = []

    for batch_x, batch_y, meta in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        probs_list.append(probs.astype(np.float32))
        pred_list.append(preds.astype(np.int64))
        label_list.append(batch_y.numpy().astype(np.int64))
        label_name_list.append(np.array(meta["label_name"]).astype(str))
        snr_list.append(np.array(meta["snr_db"]).astype(np.int16))
        channel_list.append(np.array(meta["channel_model"]).astype(str))

    result = {
        "split_name": np.array([split_name]),
        "probabilities": np.concatenate(probs_list, axis=0),
        "pred_label": np.concatenate(pred_list, axis=0),
        "true_label": np.concatenate(label_list, axis=0),
        "true_label_name": np.concatenate(label_name_list, axis=0),
        "snr_db": np.concatenate(snr_list, axis=0),
        "channel_model": np.concatenate(channel_list, axis=0),
    }
    return result


def save_history_csv(path: Path, history: List[Dict]):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "best_val_acc_so_far",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_curves(report_dir: Path, history: List[Dict]):
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    val_acc = [item["val_acc"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("1D-CNN Training / Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(report_dir / "train_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("1D-CNN Training / Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(report_dir / "train_accuracy.png", dpi=200)
    plt.close()


def evaluate_accuracy_from_prediction_npz(pred_dict: Dict[str, np.ndarray]) -> float:
    y_true = pred_dict["true_label"]
    y_pred = pred_dict["pred_label"]
    return float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    project_root = Path(args.project_root).resolve()
    model_dir = project_root / "models"
    report_dir = project_root / "reports"
    metadata_dir = project_root / "dataset" / "metadata"

    ensure_dir(model_dir)
    ensure_dir(report_dir)
    ensure_dir(metadata_dir)

    label_map = load_label_map(metadata_dir)
    idx_to_name = inverse_label_map(label_map)
    num_classes = len(label_map)

    device = resolve_device(args.device)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders(
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = OneDCNNClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_acc = -1.0
    best_epoch = -1
    early_stop_counter = 0
    history: List[Dict] = []

    best_model_path = model_dir / "best_1dcnn.pt"
    last_model_path = model_dir / "last_1dcnn.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            early_stop_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "label_map": label_map,
                "idx_to_name": idx_to_name,
                "input_shape": [2, int(train_dataset.iq.shape[-1])],
                "model_name": "1dcnn-v1",
            }
            torch.save(checkpoint, best_model_path)
        else:
            early_stop_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "label_map": label_map,
                "idx_to_name": idx_to_name,
                "input_shape": [2, int(train_dataset.iq.shape[-1])],
                "model_name": "1dcnn-v1",
            },
            last_model_path,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "train_acc": round(float(train_acc), 6),
            "val_loss": round(float(val_loss), 6),
            "val_acc": round(float(val_acc), 6),
            "best_val_acc_so_far": round(float(best_val_acc), 6),
        }
        history.append(epoch_record)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"best_val_acc={best_val_acc:.4f}, lr={current_lr:.6f}"
        )

        if early_stop_counter >= args.patience:
            print(f"[INFO] 触发 Early Stop，连续 {args.patience} 轮验证集未提升。")
            break

    if not best_model_path.exists():
        raise RuntimeError("训练结束后未保存 best_1dcnn.pt，请检查训练过程。")

    save_history_csv(report_dir / "train_history.csv", history)
    with (report_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_curves(report_dir, history)

    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    train_pred = collect_predictions(model, train_loader, device, split_name="train")
    val_pred = collect_predictions(model, val_loader, device, split_name="val")
    test_pred = collect_predictions(model, test_loader, device, split_name="test")

    np.savez_compressed(report_dir / "train_predictions.npz", **train_pred)
    np.savez_compressed(report_dir / "val_predictions.npz", **val_pred)
    np.savez_compressed(report_dir / "test_predictions.npz", **test_pred)

    summary = {
        "model_name": "1dcnn-v1",
        "device": str(device),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "best_val_acc": round(float(best_val_acc), 6),
        "train_acc_best_model": round(evaluate_accuracy_from_prediction_npz(train_pred), 6),
        "val_acc_best_model": round(evaluate_accuracy_from_prediction_npz(val_pred), 6),
        "test_acc_best_model": round(evaluate_accuracy_from_prediction_npz(test_pred), 6),
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_trained": len(history),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
    }

    with (report_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("1D-CNN 训练完成")
    print(f"best_model: {best_model_path}")
    print(f"last_model: {last_model_path}")
    print(f"history_csv: {report_dir / 'train_history.csv'}")
    print(f"loss_curve: {report_dir / 'train_loss.png'}")
    print(f"acc_curve: {report_dir / 'train_accuracy.png'}")
    print(f"train_pred: {report_dir / 'train_predictions.npz'}")
    print(f"val_pred: {report_dir / 'val_predictions.npz'}")
    print(f"test_pred: {report_dir / 'test_predictions.npz'}")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_acc: {best_val_acc:.4f}")
    print(f"test_acc_best_model: {summary['test_acc_best_model']:.4f}")

if __name__ == "__main__":
    main()