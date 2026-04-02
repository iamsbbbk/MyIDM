import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from industrial_semantic.config import load_config
from industrial_semantic.dataset import (
    SlidingWindowBuffer,
    _load_any_array,
    _split_loaded_samples,
    _ensure_image_ch_first,
)
from industrial_semantic.protocol import SemanticPacket, packet_to_bytes
from industrial_semantic.runtime import IndustrialSemanticSenderEngine, MyIDMReceiverAdapter
from industrial_semantic.utils import get_logger, compute_signal_anchors


SUPPORTED_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".csv", ".txt", ".mat"}


def iter_input_files(input_path: Path):
    if input_path.is_file():
        yield input_path
        return

    files = []
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(p)

    for p in sorted(files):
        yield p


def infer_class_hint_from_name(path: Path):
    """
    仅作为 hint，不强绑定。
    例如 rgb_x_0.npy -> 0
    """
    stem = path.stem
    parts = stem.split("_")
    for token in reversed(parts):
        if token.isdigit():
            return int(token)
    return None


def compute_image_anchors(image: np.ndarray):
    image = _ensure_image_ch_first(image)
    anchors = {
        "pixel_mean": float(np.mean(image)),
        "pixel_std": float(np.std(image)),
        "pixel_min": float(np.min(image)),
        "pixel_max": float(np.max(image)),
        "pixel_rms": float(np.sqrt(np.mean(image.astype(np.float32) ** 2) + 1e-12)),
        "spatial_tv": float(
            np.mean(np.abs(np.diff(image, axis=1))) + np.mean(np.abs(np.diff(image, axis=2)))
        ),
    }
    for i, axis_name in enumerate(["r", "g", "b"][: image.shape[0]]):
        anchors[f"{axis_name}_mean"] = float(np.mean(image[i]))
        anchors[f"{axis_name}_std"] = float(np.std(image[i]))
    return anchors


def build_header(cfg, metadata, payload_mode, data_modality):
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})

    return {
        "protocol_version": "ISC-TWM-1.1",
        "session_id": metadata.get("session_id", runtime_cfg.get("session_id", "SESSION-001")),
        "machine_id": metadata.get("machine_id", runtime_cfg.get("machine_id", "M001")),
        "spindle_id": metadata.get("spindle_id", runtime_cfg.get("spindle_id", "S001")),
        "tool_id": metadata.get("tool_id", runtime_cfg.get("tool_id", "T001")),
        "run_id": metadata.get("run_id", "RUN-001"),
        "window_id": metadata.get("window_id", "0"),
        "timestamp_start": float(metadata.get("timestamp_start", 0.0)),
        "timestamp_end": float(metadata.get("timestamp_end", 0.0)),
        "sampling_rate": int(metadata.get("sampling_rate", dataset_cfg.get("sampling_rate", 10240))),
        "window_length": int(metadata.get("window_length", dataset_cfg.get("window_length", 1024))),
        "axis_order": "XYZ" if data_modality == "signal1d" else "CHW",
        "encoder_version": runtime_cfg.get("encoder_version", "semtx-coldstart-1.0"),
        "normalization_id": metadata.get("normalization_id", cfg.get("dataset", {}).get("normalization", {}).get("id", "NA")),
        "payload_mode": payload_mode,
        "data_modality": data_modality,
        "checksum": "",
    }


def build_rawsafe_packet(cfg, sample, modality, metadata, class_hint=None):
    """
    当没有 sender checkpoint，或者当前样本不是 1D 振动窗口时，
    走 raw-safe / semantic-lite 模式。
    """
    if modality == "signal1d":
        anchors = compute_signal_anchors(sample, sampling_rate=int(cfg["dataset"]["sampling_rate"]))
    elif modality == "image2d":
        anchors = compute_image_anchors(sample)
    else:
        anchors = {}

    semantic = {
        "data_modality": modality,
        "contact_state": "unknown",
        "wear_state": "NA",
        "mapped_class": int(class_hint) if class_hint is not None else -1,
        "class_hint": int(class_hint) if class_hint is not None else -1,
        "confidence": 0.0,
        "uncertainty": 1.0,
        "ood_score": 1.0,
        "cold_start": True,
    }

    header = build_header(cfg, metadata, payload_mode="M3", data_modality=modality)

    payload = {
        "alarm": True,
        "cold_start": True,
        "data_modality": modality,
        "raw_input": np.asarray(sample, dtype=np.float32),
    }
    if modality == "signal1d":
        payload["raw_signal"] = np.asarray(sample, dtype=np.float32)
    elif modality == "image2d":
        payload["raw_image"] = np.asarray(sample, dtype=np.float32)

    return SemanticPacket(
        header=header,
        semantic=semantic,
        anchors=anchors,
        payload=payload,
    )


def expand_signal_sample_to_windows(sample: np.ndarray, window_size: int, hop_size: int, pad_tail: bool):
    """
    把单个长时序样本拆成若干窗口。
    输入 sample: [C,T]
    """
    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim != 2:
        return []

    if sample.shape[1] == window_size:
        return [sample]

    if sample.shape[1] < window_size:
        if not pad_tail:
            return []
        out = np.zeros((sample.shape[0], window_size), dtype=np.float32)
        out[:, : sample.shape[1]] = sample
        return [out]

    buffer = SlidingWindowBuffer(window_size=window_size, hop_size=hop_size, channels=sample.shape[0])
    windows = buffer.push(sample)
    if pad_tail:
        windows.extend(buffer.flush_tail(pad=True))
    return windows


def flush_sender_packets(sender, receiver, output_packet_dir, packet_counter, logger):
    flushed_packets = sender.finalize(return_bytes=False)
    for pkt in flushed_packets:
        pkt_bytes = packet_to_bytes(pkt)
        recv = receiver.receive(pkt)

        logger.info(
            f"[flush] used_path={recv.get('used_path', 'NA')} "
            f"mode={recv.get('packet_mode', 'NA')} "
            f"mapped={recv.get('semantic', {}).get('mapped_class', 'NA')}"
        )

        if output_packet_dir:
            packet_path = output_packet_dir / f"packet_{packet_counter:06d}.bin"
            with open(packet_path, "wb") as f:
                f.write(pkt_bytes)
        packet_counter += 1
    return packet_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/industrial_semantic.yaml")
    parser.add_argument("--sender_ckpt", type=str, default="", help="可选；若不存在则进入 cold-start raw-safe 模式")
    parser.add_argument("--input_path", type=str, required=True, help="单文件或目录")
    parser.add_argument("--output_packet_dir", type=str, default="", help="保存 packet 二进制目录")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pad_tail", action="store_true", help="对长时序尾部不足窗口的部分补零发送")
    parser.add_argument("--max_files", type=int, default=0, help="最多处理多少个文件；0 表示不限制")
    parser.add_argument("--max_samples_per_file", type=int, default=0, help="每个 batched 文件最多处理多少样本；0 表示不限制")
    parser.add_argument("--max_packets", type=int, default=0, help="最多生成多少 packet；0 表示不限制")
    args = parser.parse_args()

    logger = get_logger("run_semantic_demo")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    cfg = load_config(args.config)

    sender = None
    sender_ckpt_path = Path(args.sender_ckpt) if args.sender_ckpt else None
    if sender_ckpt_path is not None and str(sender_ckpt_path) != "":
        if sender_ckpt_path.exists():
            sender = IndustrialSemanticSenderEngine(
                config=cfg,
                checkpoint_path=str(sender_ckpt_path),
                device=args.device,
            )
            cfg.setdefault("receiver", {})
            if not cfg["receiver"].get("reference_checkpoint", ""):
                cfg["receiver"]["reference_checkpoint"] = str(sender_ckpt_path)
            logger.info(f"已加载 sender checkpoint: {sender_ckpt_path}")
        else:
            logger.warning("未找到 sender checkpoint，将进入 cold-start raw-safe 模式；所有不适配样本走 M3。")
    else:
        logger.warning("未提供 sender checkpoint，将进入 cold-start raw-safe 模式；所有不适配样本走 M3。")

    receiver = MyIDMReceiverAdapter(
        config=cfg,
        device=args.device,
    )

    output_packet_dir = Path(args.output_packet_dir) if args.output_packet_dir else None
    if output_packet_dir:
        output_packet_dir.mkdir(parents=True, exist_ok=True)

    window_size = int(cfg["dataset"]["window_length"])
    hop_size = int(cfg["dataset"]["window_length"])
    sr = int(cfg["dataset"]["sampling_rate"])

    packet_counter = 0
    file_count = 0
    max_files = int(args.max_files)
    max_packets = int(args.max_packets)
    max_samples_per_file = int(args.max_samples_per_file) if int(args.max_samples_per_file) > 0 else None

    for file_idx, file_path in enumerate(iter_input_files(input_path)):
        if max_files > 0 and file_count >= max_files:
            break
        if max_packets > 0 and packet_counter >= max_packets:
            break

        file_count += 1
        logger.info(f"正在处理文件: {file_path}")

        try:
            raw_arr = _load_any_array(file_path)
        except Exception as e:
            logger.warning(f"跳过文件，读取失败: {file_path}, err={e}")
            continue

        split_info = _split_loaded_samples(raw_arr, max_samples=max_samples_per_file)
        kind = split_info["kind"]
        modality = split_info["modality"]
        samples = split_info["samples"]

        logger.info(f"[inspect] shape={tuple(raw_arr.shape)} kind={kind} modality={modality}")

        # 自动跳过 label 文件
        if kind == "label_vector":
            logger.warning(f"检测到 label 向量文件，自动跳过: {file_path}, shape={raw_arr.shape}")
            continue

        if modality is None or len(samples) == 0:
            logger.warning(f"无法解析为有效样本，自动跳过: {file_path}, shape={raw_arr.shape}")
            continue

        # 切换新文件时，重置 sender 序列状态，避免跨文件污染
        if sender is not None:
            sender.reset_sequence_state()

        class_hint = infer_class_hint_from_name(file_path)

        for sample_idx, sample in enumerate(samples):
            if max_packets > 0 and packet_counter >= max_packets:
                break

            if modality == "signal1d":
                expanded_samples = expand_signal_sample_to_windows(
                    sample=sample,
                    window_size=window_size,
                    hop_size=hop_size,
                    pad_tail=args.pad_tail,
                )
                if len(expanded_samples) == 0:
                    logger.warning(
                        f"信号样本长度不足且未启用 pad_tail，跳过: file={file_path}, sample_idx={sample_idx}, shape={sample.shape}"
                    )
                    continue
            else:
                expanded_samples = [sample]

            for sub_idx, item in enumerate(expanded_samples):
                if max_packets > 0 and packet_counter >= max_packets:
                    break

                metadata = {
                    "run_id": file_path.parent.name,
                    "tool_id": file_path.parent.parent.name if file_path.parent.parent != file_path.parent else "T001",
                    "window_id": f"{file_idx}_{sample_idx}_{sub_idx}",
                    "timestamp_start": float(sub_idx * window_size / sr) if modality == "signal1d" else float(sample_idx),
                    "timestamp_end": float((sub_idx + 1) * window_size / sr) if modality == "signal1d" else float(sample_idx + 1),
                    "sampling_rate": sr,
                    "window_length": window_size if modality == "signal1d" else int(item.shape[-1]) if item.ndim >= 2 else 0,
                    "normalization_id": cfg.get("dataset", {}).get("normalization", {}).get("id", "NA"),
                }

                packets = []

                # 只有 sender 存在且当前是 1D signal 时，才走工业语义 sender
                if sender is not None and modality == "signal1d" and item.ndim == 2 and item.shape[1] == window_size:
                    result = sender.process_window(item, metadata=metadata, return_bytes=False)
                    analysis = result["analysis"]
                    packets = result["packets"]

                    logger.info(
                        f"[analysis] mode={analysis['mode']} "
                        f"contact={analysis['semantic']['contact_state']} "
                        f"wear={analysis['semantic']['wear_state']} "
                        f"mapped={analysis['semantic']['mapped_class']} "
                        f"conf={analysis['semantic']['confidence']:.4f} "
                        f"unc={analysis['semantic']['uncertainty']:.4f} "
                        f"occ={analysis['semantic']['occupancy_ratio']:.2f}"
                    )
                else:
                    # 图像模态，或没有 sender ckpt，统一走 raw-safe 模式
                    pkt = build_rawsafe_packet(
                        cfg=cfg,
                        sample=item,
                        modality=modality,
                        metadata=metadata,
                        class_hint=class_hint,
                    )
                    packets = [pkt]

                    reason = "cold-start" if sender is None else f"modality_bypass:{modality}"
                    logger.info(
                        f"[raw-safe] reason={reason} "
                        f"modality={modality} "
                        f"shape={tuple(item.shape)} "
                        f"class_hint={class_hint}"
                    )

                for pkt in packets:
                    pkt_bytes = packet_to_bytes(pkt)
                    recv = receiver.receive(pkt)

                    logger.info(
                        f"[recv] used_path={recv.get('used_path', 'NA')} "
                        f"mode={recv.get('packet_mode', 'NA')} "
                        f"mapped={recv.get('semantic', {}).get('mapped_class', 'NA')} "
                        f"modality={recv.get('data_modality', recv.get('semantic', {}).get('data_modality', 'NA'))}"
                    )

                    if output_packet_dir:
                        packet_path = output_packet_dir / f"packet_{packet_counter:06d}.bin"
                        with open(packet_path, "wb") as f:
                            f.write(pkt_bytes)

                    packet_counter += 1

        # 对 signal sender 做文件级 flush
        if sender is not None:
            packet_counter = flush_sender_packets(
                sender=sender,
                receiver=receiver,
                output_packet_dir=output_packet_dir,
                packet_counter=packet_counter,
                logger=logger,
            )

    logger.info(f"演示完成，共处理 {file_count} 个文件，生成 {packet_counter} 个 packet。")


if __name__ == "__main__":
    main()