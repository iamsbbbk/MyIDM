from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def load_registry(registry_path: str | None, checkpoints_root: str | None):
    if registry_path:
        p = Path(registry_path)
        if not p.exists():
            raise FileNotFoundError(f"registry 不存在: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    if checkpoints_root:
        p = Path(checkpoints_root) / "deploy_registry.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))

    return None


def resolve_checkpoint(cls: int, checkpoints_root: str, registry: dict | None):
    if registry is not None:
        entry = registry.get("classes", {}).get(str(cls), {})
        ckpt = entry.get("checkpoint", None)
        if ckpt:
            p = Path(ckpt)
            if p.exists():
                return p

    root = Path(checkpoints_root)
    p = root / f"class{cls}" / f"best_model_cls{cls}.pth"
    if p.exists():
        return p

    p2 = root / f"best_model_cls{cls}.pth"
    if p2.exists():
        return p2

    raise FileNotFoundError(f"未找到 class={cls} 的 checkpoint")


def resolve_device(devices, idx):
    if not devices:
        return None
    if len(devices) == 1:
        return devices[0]
    if len(devices) != 0 and len(devices) != idx["total"]:
        raise ValueError("devices 数量必须为 1 或与 classes 数量一致")
    return devices[idx["i"]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--classes", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--checkpoints_root", type=str, default="./checkpoints_4cls")
    parser.add_argument("--registry", type=str, default=None)

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--base_port", type=int, default=19001)
    parser.add_argument("--secret", type=str, default="demo-secret")

    parser.add_argument("--sd_path", type=str, default="./sd15")
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--step_number", type=int, default=None)
    parser.add_argument("--devices", nargs="*", default=None, help="如: cuda:0 cuda:1 cuda:2 cuda:3")
    parser.add_argument("--use_amp", type=str2bool, default=None)

    parser.add_argument("--log_root", type=str, default=None)
    parser.add_argument("--hf_endpoint", type=str, default="")

    args = parser.parse_args()

    registry = load_registry(args.registry, args.checkpoints_root)

    log_root = Path(args.log_root) if args.log_root else Path(args.checkpoints_root) / "service_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    procs = []

    try:
        total = len(args.classes)
        for i, cls in enumerate(args.classes):
            ckpt = resolve_checkpoint(cls, args.checkpoints_root, registry)
            port = args.base_port + i

            if not args.devices:
                device = None
            elif len(args.devices) == 1:
                device = args.devices[0]
            elif len(args.devices) == total:
                device = args.devices[i]
            else:
                raise ValueError("devices 数量必须为 1 或与 classes 数量一致")

            cmd = [
                sys.executable,
                "-m",
                "comm.receiver_server",
                "--adapter", "myidm",
                "--host", str(args.host),
                "--port", str(port),
                "--secret", str(args.secret),
                "--checkpoint", str(ckpt),
                "--sd-path", str(args.sd_path),
                "--block-size", str(args.block_size),
                "--max-inflight", "1",
            ]

            if args.step_number is not None:
                cmd += ["--step-number", str(args.step_number)]
            if device is not None:
                cmd += ["--device", str(device)]
            if args.use_amp is not None:
                cmd += ["--use-amp", str(args.use_amp)]

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if args.hf_endpoint:
                env["HF_ENDPOINT"] = str(args.hf_endpoint)

            log_path = log_root / f"receiver_cls{cls}.log"
            f = open(log_path, "w", encoding="utf-8")
            p = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

            procs.append((p, f, cls, port, ckpt, log_path, device))

            print(
                f"[START] class={cls} | port={port} | ckpt={ckpt} | "
                f"device={device} | log={log_path}"
            )

        print("\n全部接收端已启动。")
        print("端口映射：")
        for i, cls in enumerate(args.classes):
            print(f"  class={cls} -> {args.base_port + i}")

        print("\n按 Ctrl+C 停止全部服务。\n")

        while True:
            time.sleep(3)
            for p, f, cls, port, ckpt, log_path, device in procs:
                rc = p.poll()
                if rc is not None:
                    raise RuntimeError(
                        f"class={cls} 的接收端提前退出，rc={rc}，请检查日志: {log_path}"
                    )

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止全部服务...")
    finally:
        for p, f, cls, port, ckpt, log_path, device in procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass

        time.sleep(2)

        for p, f, cls, port, ckpt, log_path, device in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass

        print("全部接收端已停止。")


if __name__ == "__main__":
    main()