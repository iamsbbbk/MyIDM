import argparse
from pathlib import Path

import numpy as np

from industrial_semantic.config import load_config
from industrial_semantic.receiver import IndustrialSemanticReceiverEngine
from industrial_semantic.utils import get_logger, save_json


def iter_packet_files(path: Path, suffix: str = ".bin"):
    if path.is_file():
        yield path
        return

    files = sorted([p for p in path.rglob(f"*{suffix}") if p.is_file()])
    for p in files:
        yield p


def prune_large_arrays(obj):
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray_summary__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "nbytes": int(obj.nbytes),
        }
    if isinstance(obj, dict):
        return {k: prune_large_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [prune_large_arrays(v) for v in obj]
    if isinstance(obj, tuple):
        return [prune_large_arrays(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/industrial_semantic.yaml")
    parser.add_argument("--packet_path", type=str, required=True, help="单个 packet 文件或目录")
    parser.add_argument("--output_json_dir", type=str, default="", help="保存 receiver 结果 json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("run_semantic_receiver")

    engine = IndustrialSemanticReceiverEngine(
        config=cfg,
        device=args.device,
    )

    packet_path = Path(args.packet_path)
    output_json_dir = Path(args.output_json_dir) if args.output_json_dir else None
    if output_json_dir:
        output_json_dir.mkdir(parents=True, exist_ok=True)

    mode_hist = {}
    path_hist = {}
    alarm_count = 0
    total = 0

    for idx, packet_file in enumerate(iter_packet_files(packet_path)):
        result = engine.receive_file(packet_file)

        mode = result.get("packet_mode", "NA")
        used_path = result.get("used_path", "unknown")

        mode_hist[mode] = mode_hist.get(mode, 0) + 1
        path_hist[used_path] = path_hist.get(used_path, 0) + 1
        if bool(result.get("alarm", False)):
            alarm_count += 1
        total += 1

        logger.info(
            f"[{idx:06d}] file={packet_file.name} "
            f"mode={mode} "
            f"used_path={used_path} "
            f"mapped={result.get('semantic', {}).get('mapped_class', 'NA')} "
            f"alarm={result.get('alarm', False)} "
            f"status={result.get('status', 'ok')}"
        )

        if output_json_dir:
            pruned = prune_large_arrays(result)
            save_json(pruned, output_json_dir / f"{packet_file.stem}.json")

    logger.info(f"共处理 packet 数量: {total}")
    logger.info(f"mode 分布: {mode_hist}")
    logger.info(f"used_path 分布: {path_hist}")
    logger.info(f"alarm 数量: {alarm_count}")


if __name__ == "__main__":
    main()