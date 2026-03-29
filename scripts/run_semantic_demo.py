import argparse
import sys
from pathlib import Path

# ---------------------------
# 关键修复：把项目根目录加入 sys.path
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from industrial_semantic.config import load_config
from industrial_semantic.dataset import SlidingWindowBuffer, _load_signal
from industrial_semantic.protocol import packet_to_bytes
from industrial_semantic.runtime import IndustrialSemanticSenderEngine, MyIDMReceiverAdapter
from industrial_semantic.utils import get_logger


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
    parser.add_argument("--sender_ckpt", type=str, required=True, help="发送端 checkpoint，如 best.pt")
    parser.add_argument("--input_path", type=str, required=True, help="单文件或目录")
    parser.add_argument("--output_packet_dir", type=str, default="", help="保存 packet 二进制目录")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pad_tail", action="store_true", help="是否对尾部不足一个窗口的数据补零发送")
    args = parser.parse_args()

    logger = get_logger("run_semantic_demo")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    sender_ckpt = Path(args.sender_ckpt)
    if not sender_ckpt.exists():
        raise FileNotFoundError(f"sender checkpoint 不存在: {sender_ckpt}")

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"输入路径不存在: {input_path}\n"
            f"注意：你现在传入的是占位符 './your_toolwear_data'，请替换成真实数据路径。"
        )

    cfg = load_config(args.config)

    # 自动把 receiver 的 reference_checkpoint 指向 sender ckpt，保证 fallback 可用
    cfg.setdefault("receiver", {})
    if not cfg["receiver"].get("reference_checkpoint", ""):
        cfg["receiver"]["reference_checkpoint"] = str(sender_ckpt)

    sender = IndustrialSemanticSenderEngine(
        config=cfg,
        checkpoint_path=str(sender_ckpt),
        device=args.device,
    )
    receiver = MyIDMReceiverAdapter(
        config=cfg,
        device=args.device,
    )

    output_packet_dir = Path(args.output_packet_dir) if args.output_packet_dir else None
    if output_packet_dir:
        output_packet_dir.mkdir(parents=True, exist_ok=True)

    window_size = int(cfg["dataset"]["window_length"])
    hop_size = int(cfg["dataset"]["window_length"])

    packet_counter = 0
    file_count = 0

    for file_idx, file_path in enumerate(iter_input_files(input_path)):
        file_count += 1
        logger.info(f"处理文件: {file_path}")

        # 非常重要：切换文件时重置 sender 的序列状态，避免状态跨文件污染
        sender.reset_sequence_state()

        buffer = SlidingWindowBuffer(window_size=window_size, hop_size=hop_size, channels=3)
        signal = _load_signal(file_path).astype(np.float32)  # [C, T]

        windows = buffer.push(signal)
        if args.pad_tail:
            windows.extend(buffer.flush_tail(pad=True))

        if len(windows) == 0 and signal.shape[1] == window_size:
            windows = [signal]

        sr = int(cfg["dataset"]["sampling_rate"])

        for w_idx, win in enumerate(windows):
            metadata = {
                "run_id": file_path.parent.name,
                "tool_id": file_path.parent.parent.name if file_path.parent.parent != file_path.parent else "T001",
                "window_id": f"{file_idx}_{w_idx}",
                "timestamp_start": float(w_idx * window_size / sr),
                "timestamp_end": float((w_idx + 1) * window_size / sr),
                "sampling_rate": sr,
                "window_length": window_size,
            }

            result = sender.process_window(win, metadata=metadata, return_bytes=False)
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

            for pkt in packets:
                pkt_bytes = packet_to_bytes(pkt)
                recv = receiver.receive(pkt)

                logger.info(
                    f"[recv] used_path={recv.get('used_path', 'NA')} "
                    f"mode={recv.get('packet_mode', 'NA')} "
                    f"mapped={recv.get('semantic', {}).get('mapped_class', 'NA')}"
                )

                if output_packet_dir:
                    packet_path = output_packet_dir / f"packet_{packet_counter:06d}.bin"
                    with open(packet_path, "wb") as f:
                        f.write(pkt_bytes)
                packet_counter += 1

        # 每个文件结束都 flush，防止 idle 段跨文件串联
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