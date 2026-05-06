"""
LSTM 调制识别对比模型
"""
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


LABEL_MAP = {"AM": 0, "FM": 1, "BPSK": 2, "QPSK": 3, "16QAM": 4}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_label_map(project_root: Path) -> Dict[str, int]:
    path = project_root / "dataset" / "metadata" / "label_map.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return LABEL_MAP


class IQDataset(Dataset):
    def __init__(self, npz_path: Path, normalize: bool = True):
        data = np.load(npz_path, allow_pickle=False)
        self.iq = data["iq"].astype(np.float32)       # [N, 2, L]
        self.label = data["label"].astype(np.int64)   # [N]
        self.normalize = normalize
        if self.iq.ndim != 3 or self.iq.shape[1] != 2:
            raise ValueError(f"iq 应为 [N, 2, L]，实际为 {self.iq.shape}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx: int):
        x = self.iq[idx].copy()
        if self.normalize:
            for c in range(x.shape[0]):
                x[c] = (x[c] - x[c].mean()) / (x[c].std() + 1e-6)
        y = self.label[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class LSTMClassifier(nn.Module):
    """简单 LSTM 调制识别模型。输入 [B, 2, L]，内部转换为 [B, L, 2]。"""
    def __init__(self, num_classes: int = 5, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # [B, L, 2]
        out, _ = self.lstm(x)          # [B, L, H]
        last_feature = out[:, -1, :]   # 取最后一个时刻特征
        return self.classifier(last_feature)


def build_loaders(project_root: Path, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root = project_root / "dataset"
    train_set = IQDataset(root / "train" / "train.npz")
    val_set = IQDataset(root / "val" / "val.npz")
    test_set = IQDataset(root / "test" / "test.npz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total_count = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += y.size(0)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser(description="训练 LSTM 调制识别对比模型")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260325)
    args = parser.parse_args()

    set_seed(args.seed)
    project_root = Path(args.project_root).resolve()
    label_map = load_label_map(project_root)
    num_classes = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_loaders(project_root, args.batch_size)
    model = LSTMClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_dir = project_root / "models"
    report_dir = project_root / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    history = []
    best_path = model_dir / "best_lstm.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        })
        print(f"[LSTM] epoch={epoch:02d} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": "lstm-v1",
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "input_shape": [2, 256],
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

    summary = {
        "model_name": "lstm-v1",
        "device": str(device),
        "best_val_acc": round(best_val_acc, 6),
        "test_acc": round(test_acc, 6),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    }
    with (report_dir / "lstm_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (report_dir / "lstm_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
