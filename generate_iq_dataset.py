import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


LABEL_MAP = {
    "AM": 0,
    "FM": 1,
    "BPSK": 2,
    "QPSK": 3,
    "16QAM": 4,
}

CHANNEL_MODELS = [
    "AWGN",
    "Rayleigh",
    "CarrierOffset",
    "SampleRateError",
    "PathLoss",
]


def set_seed(seed: int):
    np.random.seed(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_signal(x: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    """
    归一化复信号平均功率
    """
    power = np.mean(np.abs(x) ** 2) + 1e-12
    scale = math.sqrt(target_power / power)
    return x * scale


def add_awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    """
    按目标 SNR 添加复高斯白噪声
    """
    signal_power = np.mean(np.abs(x) ** 2) + 1e-12
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / (snr_linear + 1e-12)

    noise = (
        np.random.randn(len(x)) + 1j * np.random.randn(len(x))
    ) * math.sqrt(noise_power / 2.0)

    return x + noise


def apply_rayleigh_fading(x: np.ndarray) -> np.ndarray:
    """
    简化版瑞利衰落：构造慢变复衰落包络
    """
    n = len(x)
    knots = max(6, n // 32)

    real_knots = np.random.randn(knots)
    imag_knots = np.random.randn(knots)

    knot_pos = np.linspace(0, n - 1, knots)
    pos = np.arange(n)

    real_interp = np.interp(pos, knot_pos, real_knots)
    imag_interp = np.interp(pos, knot_pos, imag_knots)

    h = real_interp + 1j * imag_interp
    h = normalize_signal(h, target_power=1.0)

    return x * h


def apply_carrier_offset(x: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    简化版载波频偏
    """
    n = len(x)
    t = np.arange(n) / sample_rate
    freq_offset = np.random.uniform(-0.03, 0.03) * sample_rate
    phase = 2 * np.pi * freq_offset * t
    return x * np.exp(1j * phase)


def apply_sample_rate_error(x: np.ndarray) -> np.ndarray:
    """
    简化版采样率误差：插值重采样后再裁剪/补齐
    """
    n = len(x)
    error_ratio = np.random.uniform(-0.02, 0.02)

    old_pos = np.arange(n)
    new_pos = np.arange(n) * (1.0 + error_ratio)

    real_part = np.interp(old_pos, np.clip(new_pos, 0, n - 1), np.real(x))
    imag_part = np.interp(old_pos, np.clip(new_pos, 0, n - 1), np.imag(x))

    return real_part + 1j * imag_part


def apply_path_loss(x: np.ndarray) -> np.ndarray:
    """
    简化版路损：随机幅度衰减 + 轻微慢变
    """
    n = len(x)
    base_loss = np.random.uniform(0.25, 0.85)

    knots = max(4, n // 64)
    knot_vals = np.random.uniform(0.9, 1.1, size=knots)
    knot_pos = np.linspace(0, n - 1, knots)
    pos = np.arange(n)
    envelope = np.interp(pos, knot_pos, knot_vals)

    return x * base_loss * envelope


def smooth_signal(x: np.ndarray) -> np.ndarray:
    """
    轻量脉冲整形/平滑
    """
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    real_s = np.convolve(np.real(x), kernel, mode="same")
    imag_s = np.convolve(np.imag(x), kernel, mode="same")
    return real_s + 1j * imag_s


def generate_random_message(seq_len: int) -> np.ndarray:
    """
    生成低频随机消息，用于 AM / FM
    """
    t = np.linspace(0, 1, seq_len, endpoint=False)
    msg = np.zeros(seq_len, dtype=np.float32)

    tone_count = np.random.randint(2, 5)
    for _ in range(tone_count):
        freq = np.random.uniform(1.0, 8.0)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.2, 1.0)
        msg += amp * np.sin(2 * np.pi * freq * t + phase)

    msg = msg / (np.max(np.abs(msg)) + 1e-12)
    return msg.astype(np.float32)


def generate_am(seq_len: int, sample_rate: float) -> np.ndarray:
    t = np.arange(seq_len) / sample_rate
    msg = generate_random_message(seq_len)
    modulation_index = np.random.uniform(0.3, 0.9)
    carrier_freq = np.random.uniform(0.04, 0.12) * sample_rate

    envelope = 1.0 + modulation_index * msg
    phase = 2 * np.pi * carrier_freq * t + np.random.uniform(0, 2 * np.pi)

    x = envelope * np.exp(1j * phase)
    return normalize_signal(x)


def generate_fm(seq_len: int, sample_rate: float) -> np.ndarray:
    msg = generate_random_message(seq_len)
    freq_dev = np.random.uniform(0.01, 0.06) * sample_rate
    carrier_freq = np.random.uniform(0.03, 0.10) * sample_rate

    instantaneous_freq = carrier_freq + freq_dev * msg
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate

    x = np.exp(1j * phase)
    return normalize_signal(x)


def qam16_constellation():
    points = np.array(
        [
            -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
            -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
             1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,
             3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j,
        ],
        dtype=np.complex64
    )
    points = points / np.sqrt(np.mean(np.abs(points) ** 2))
    return points


def generate_digital_signal(mod_type: str, seq_len: int) -> np.ndarray:
    """
    简化版数字调制：
    随机符号 -> 重复上采样 -> 平滑
    """
    sps = np.random.choice([4, 8, 16])
    symbol_count = math.ceil(seq_len / sps) + 2

    if mod_type == "BPSK":
        symbols = np.random.choice([-1, 1], size=symbol_count).astype(np.float32)
        x = symbols.astype(np.complex64)

    elif mod_type == "QPSK":
        bits_i = np.random.choice([-1, 1], size=symbol_count)
        bits_q = np.random.choice([-1, 1], size=symbol_count)
        x = (bits_i + 1j * bits_q) / np.sqrt(2)

    elif mod_type == "16QAM":
        const = qam16_constellation()
        idx = np.random.randint(0, len(const), size=symbol_count)
        x = const[idx]

    else:
        raise ValueError(f"不支持的数字调制类型: {mod_type}")

    upsampled = np.repeat(x, sps)[:seq_len]
    if len(upsampled) < seq_len:
        pad_len = seq_len - len(upsampled)
        upsampled = np.pad(upsampled, (0, pad_len), mode="edge")

    upsampled = smooth_signal(upsampled)
    return normalize_signal(upsampled)


def generate_clean_signal(label_name: str, seq_len: int, sample_rate: float) -> np.ndarray:
    if label_name == "AM":
        return generate_am(seq_len, sample_rate)
    if label_name == "FM":
        return generate_fm(seq_len, sample_rate)
    if label_name in {"BPSK", "QPSK", "16QAM"}:
        return generate_digital_signal(label_name, seq_len)
    raise ValueError(f"未知标签: {label_name}")


def apply_channel_effect(x: np.ndarray, channel_model: str, sample_rate: float) -> np.ndarray:
    y = np.array(x, copy=True)

    if channel_model == "Rayleigh":
        y = apply_rayleigh_fading(y)
    elif channel_model == "CarrierOffset":
        y = apply_carrier_offset(y, sample_rate)
    elif channel_model == "SampleRateError":
        y = apply_sample_rate_error(y)
    elif channel_model == "PathLoss":
        y = apply_path_loss(y)
    elif channel_model == "AWGN":
        pass
    else:
        raise ValueError(f"未知信道模型: {channel_model}")

    return normalize_signal(y)


def sample_to_iq_tensor(x: np.ndarray) -> np.ndarray:
    """
    转成 [2, seq_len]
    """
    iq = np.stack([np.real(x), np.imag(x)], axis=0).astype(np.float32)
    return iq


def write_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_summary_csv(path: Path, rows: list):
    fieldnames = [
        "file_name",
        "label_name",
        "label_id",
        "snr_db",
        "sample_count",
        "seq_len",
        "sample_rate",
        "channel_models",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="生成无线电调制识别 I/Q 数据集")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="ai-research 根目录，默认自动推断"
    )
    parser.add_argument("--samples-per-combination", type=int, default=500, help="每个 类别×SNR 组合生成多少条样本")
    parser.add_argument("--seq-len", type=int, default=256, help="每条 I/Q 序列长度")
    parser.add_argument("--sample-rate", type=float, default=256.0, help="采样率")
    parser.add_argument("--snr-start", type=int, default=-20, help="SNR 起始值")
    parser.add_argument("--snr-end", type=int, default=20, help="SNR 结束值")
    parser.add_argument("--snr-step", type=int, default=2, help="SNR 步长")
    parser.add_argument("--seed", type=int, default=20260325, help="随机种子")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    project_root = Path(args.project_root).resolve()
    dataset_root = project_root / "dataset"
    raw_dir = dataset_root / "raw"
    metadata_dir = dataset_root / "metadata"

    ensure_dir(raw_dir)
    ensure_dir(metadata_dir)

    snr_values = list(range(args.snr_start, args.snr_end + 1, args.snr_step))

    generation_config = {
        "labels": LABEL_MAP,
        "channel_models": CHANNEL_MODELS,
        "samples_per_combination": args.samples_per_combination,
        "seq_len": args.seq_len,
        "sample_rate": args.sample_rate,
        "snr_values": snr_values,
        "seed": args.seed,
        "description": "最小可开工版 I/Q 数据集，支持 AM/FM/BPSK/QPSK/16QAM 与多种简化信道效应",
    }

    write_json(metadata_dir / "label_map.json", LABEL_MAP)
    write_json(metadata_dir / "generation_config.json", generation_config)

    summary_rows = []
    total_samples = 0

    for label_name, label_id in LABEL_MAP.items():
        for snr_db in snr_values:
            file_name = f"{label_name}_snr_{snr_db}.npz"
            file_path = raw_dir / file_name

            iq_samples = np.zeros(
                (args.samples_per_combination, 2, args.seq_len),
                dtype=np.float32
            )
            labels = np.full((args.samples_per_combination,), label_id, dtype=np.int64)
            snrs = np.full((args.samples_per_combination,), snr_db, dtype=np.int16)
            label_names = np.array(
                [label_name] * args.samples_per_combination,
                dtype="<U16"
            )
            channel_names = np.empty((args.samples_per_combination,), dtype="<U32")

            for i in range(args.samples_per_combination):
                channel_model = np.random.choice(CHANNEL_MODELS)

                clean = generate_clean_signal(
                    label_name=label_name,
                    seq_len=args.seq_len,
                    sample_rate=args.sample_rate
                )

                with_channel = apply_channel_effect(
                    clean,
                    channel_model=channel_model,
                    sample_rate=args.sample_rate
                )

                noisy = add_awgn(with_channel, snr_db=snr_db)
                noisy = normalize_signal(noisy)

                iq_samples[i] = sample_to_iq_tensor(noisy)
                channel_names[i] = channel_model

            np.savez_compressed(
                file_path,
                iq=iq_samples,
                label=labels,
                label_name=label_names,
                snr_db=snrs,
                channel_model=channel_names,
                seq_len=np.array([args.seq_len], dtype=np.int32),
                sample_rate=np.array([args.sample_rate], dtype=np.float32),
            )

            total_samples += args.samples_per_combination

            summary_rows.append({
                "file_name": file_name,
                "label_name": label_name,
                "label_id": label_id,
                "snr_db": snr_db,
                "sample_count": args.samples_per_combination,
                "seq_len": args.seq_len,
                "sample_rate": args.sample_rate,
                "channel_models": "|".join(CHANNEL_MODELS),
            })

            print(f"[OK] 已生成 {file_name}，样本数={args.samples_per_combination}")

    write_summary_csv(metadata_dir / "raw_summary.csv", summary_rows)

    final_summary = {
        "total_files": len(summary_rows),
        "total_samples": total_samples,
        "labels": list(LABEL_MAP.keys()),
        "snr_values": snr_values,
        "samples_per_combination": args.samples_per_combination,
        "seq_len": args.seq_len,
    }
    write_json(metadata_dir / "raw_summary.json", final_summary)

    print("I/Q 原始数据集生成完成")
    print(f"project_root: {project_root}")
    print(f"raw_dir: {raw_dir}")
    print(f"metadata_dir: {metadata_dir}")
    print(f"总文件数: {len(summary_rows)}")
    print(f"总样本数: {total_samples}")

if __name__ == "__main__":
    main()