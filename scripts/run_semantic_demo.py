import argparse
from pathlib import Path

import numpy as np

from industrial_semantic.config import load_config
from industrial_semantic.dataset import SlidingWindowBuffer, _load_signal
from industrial_semantic.protocol import packet_to_bytes
from industrial_semantic.runtime import IndustrialSemanticSenderEngine
from industrial_semantic.receiver import IndustrialSemanticReceiverEngine
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/industrial_semantic.yaml")
    parser.add_argument("--sender_ckpt", type=str, required=True, help="发送端模型 checkpoint")
    parser.add_argument("--input_path", type=str, required=True, help="单文件或目录")
    parser.add_argument("--output_packet_dir", type=str, default="", help="保存 packet 二进制")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("run_semantic_demo")

    sender = IndustrialSemanticSenderEngine(
        config=cfg,
        checkpoint_path=args.sender_ckpt,
        device=args.device,
    )
    receiver = IndustrialSemanticReceiverEngine(
        config=cfg,
        device=args.device,
    )

    input_path = Path(args.input_path)
    output_packet_dir = Path(args.output_packet_dir) if args.output_packet_dir else None
    if output_packet_dir:
        output_packet_dir.mkdir(parents=True, exist_ok=True)

    window_size = int(cfg["dataset"]["window_length"])
    hop_size = int(cfg["dataset"]["window_length"])
    sampling_rate = int(cfg["dataset"]["sampling_rate"])
    buffer = SlidingWindowBuffer(window_size=window_size, hop_size=hop_size, channels=3)

    packet_counter = 0
    used_path_hist = {}

    for file_idx, file_path in enumerate(iter_input_files(input_path)):
        logger.info(f"处理文件: {file_path}")
        signal = _load_signal(file_path).astype(np.float32)  # [C, T]
        buffer.reset()
        windows = buffer.push(signal)

        if len(windows) == 0 and signal.shape[1] == window_size:
            windows = [signal]

        for w_idx, win in enumerate(windows):
            metadata = {
                "run_id": file_path.parent.name,
                "tool_id": file_path.parent.parent.name if file_path.parent.parent != file_path.parent else "T001",
                "window_id": f"{file_idx}_{w_idx}",
                "timestamp_start": float(w_idx * window_size / sampling_rate),
                "timestamp_end": float((w_idx + 1) * window_size / sampling_rate),
            }

            result = sender.process_window(win, metadata=metadata, return_bytes=False)
            analysis = result["analysis"]
            packets = result["packets"]

            logger.info(
                f"[sender] mode={analysis['mode']} "
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

                used_path = recv.get("used_path", "unknown")
                used_path_hist[used_path] = used_path_hist.get(used_path, 0) + 1

                logger.info(
                    f"[receiver] used_path={used_path} "
                    f"mode={recv['packet_mode']} "
                    f"mapped={recv['semantic'].get('mapped_class', 'NA')} "
                    f"alarm={recv.get('alarm', False)}"
                )

                if output_packet_dir:
                    packet_path = output_packet_dir / f"packet_{packet_counter:06d}.bin"
                    with open(packet_path, "wb") as f:
                        f.write(pkt_bytes)
                packet_counter += 1

    flushed_packets = sender.finalize(return_bytes=False)
    for pkt in flushed_packets:
        pkt_bytes = packet_to_bytes(pkt)
        recv = receiver.receive(pkt)

        used_path = recv.get("used_path", "unknown")
        used_path_hist[used_path] = used_path_hist.get(used_path, 0) + 1

        logger.info(
            f"[final_flush] used_path={used_path} "
            f"mode={recv['packet_mode']} "
            f"mapped={recv['semantic'].get('mapped_class', 'NA')} "
            f"alarm={recv.get('alarm', False)}"
        )
        if output_packet_dir:
            packet_path = output_packet_dir / f"packet_{packet_counter:06d}.bin"
            with open(packet_path, "wb") as f:
                f.write(pkt_bytes)
        packet_counter += 1

    logger.info(f"演示完成，共生成 {packet_counter} 个 packet")
    logger.info(f"receiver used_path 分布: {used_path_hist}")


if __name__ == "__main__":
    main()