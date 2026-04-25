from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "train.py"


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def export_phi_from_checkpoint(checkpoint: Path, out_dir: Path):
    ckpt = torch.load(checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"checkpoint 格式错误: {checkpoint}")

    msd = ckpt.get("matrix_state_dict", {})
    if "Phi" not in msd:
        raise RuntimeError(f"checkpoint 中没有 matrix_state_dict['Phi']: {checkpoint}")

    Phi = msd["Phi"].detach().float().cpu().numpy()
    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}

    out_dir.mkdir(parents=True, exist_ok=True)
    phi_path = out_dir / "Phi.npy"
    manifest_path = out_dir / "manifest.json"

    np.save(phi_path, Phi)

    manifest = {
        "checkpoint": str(checkpoint),
        "phi_shape": list(Phi.shape),
        "step_number": cfg.get("step_number", None),
        "block_size": cfg.get("block_size", None),
        "cs_ratio": cfg.get("cs_ratio", None),
        "target_class": cfg.get("target_class", None),
        "sd_path": cfg.get("sd_path", None),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return phi_path, manifest_path


def build_train_cmd(cls: int, args, save_dir: Path):
    return [
        sys.executable,
        str(TRAIN_PY),
        "--data_dir", str(args.data_dir),
        "--save_dir", str(save_dir),
        "--sd_path", str(args.sd_path),
        "--target_class", str(cls),
        "--epoch", str(args.epoch),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--step_number", str(args.step_number),
        "--cs_ratio", str(args.cs_ratio),
        "--block_size", str(args.block_size),
    ]


def build_env(args, visible_device=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = str(args.hf_endpoint)

    if visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    elif args.visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.visible_devices)

    return env


def prepare_tasks(args):
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for idx, cls in enumerate(args.classes):
        save_dir = save_root / f"class{cls}"
        save_dir.mkdir(parents=True, exist_ok=True)

        task = {
            "idx": idx,
            "cls": int(cls),
            "save_dir": save_dir,
            "log_path": save_dir / f"train_cls{cls}.log",
            "ckpt_path": save_dir / f"best_model_cls{cls}.pth",
            "cmd": build_train_cmd(int(cls), args, save_dir),
        }
        tasks.append(task)
    return tasks


def run_serial(tasks, args):
    results = []

    for task in tasks:
        cls = task["cls"]

        if args.skip_existing and task["ckpt_path"].exists():
            print(f"[SKIP] class={cls}, 已存在: {task['ckpt_path']}")
            results.append({
                "class": cls,
                "status": "skipped",
                "returncode": 0,
                "checkpoint": str(task["ckpt_path"]),
                "log_path": str(task["log_path"]),
            })
            continue

        env = build_env(args, visible_device=None)

        print(f"\n[TRAIN] class={cls}")
        print("save_dir =", task["save_dir"])
        print("log_path =", task["log_path"])
        print("cmd      =", " ".join(task["cmd"]))

        with open(task["log_path"], "w", encoding="utf-8") as f:
            proc = subprocess.run(
                task["cmd"],
                cwd=REPO_ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        result = {
            "class": cls,
            "status": "trained" if proc.returncode == 0 else "failed",
            "returncode": int(proc.returncode),
            "checkpoint": str(task["ckpt_path"]) if task["ckpt_path"].exists() else None,
            "log_path": str(task["log_path"]),
        }
        results.append(result)

        if proc.returncode != 0 and not args.continue_on_error:
            raise RuntimeError(f"class={cls} 训练失败，日志见: {task['log_path']}")

    return results


def run_parallel(tasks, args):
    if not args.devices:
        raise ValueError("--parallel true 时必须提供 --devices，例如: --devices 0 1 2 3")

    run_tasks = []
    results = []

    for task in tasks:
        cls = task["cls"]
        if args.skip_existing and task["ckpt_path"].exists():
            print(f"[SKIP] class={cls}, 已存在: {task['ckpt_path']}")
            results.append({
                "class": cls,
                "status": "skipped",
                "returncode": 0,
                "checkpoint": str(task["ckpt_path"]),
                "log_path": str(task["log_path"]),
            })
        else:
            run_tasks.append(task)

    if not run_tasks:
        return results

    if len(args.devices) < len(run_tasks):
        raise ValueError(
            f"并行模式下 devices 数量不足: devices={args.devices}, run_tasks={len(run_tasks)}"
        )

    procs = []
    try:
        for i, task in enumerate(run_tasks):
            cls = task["cls"]
            dev = args.devices[i]
            env = build_env(args, visible_device=dev)

            print(f"\n[TRAIN-PARALLEL] class={cls} on CUDA_VISIBLE_DEVICES={dev}")
            print("save_dir =", task["save_dir"])
            print("log_path =", task["log_path"])
            print("cmd      =", " ".join(task["cmd"]))

            f = open(task["log_path"], "w", encoding="utf-8")
            p = subprocess.Popen(
                task["cmd"],
                cwd=REPO_ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            procs.append((p, f, task, dev))

        while procs:
            time.sleep(5)
            for item in procs[:]:
                p, f, task, dev = item
                rc = p.poll()
                if rc is None:
                    continue

                f.close()
                procs.remove(item)

                result = {
                    "class": task["cls"],
                    "status": "trained" if rc == 0 else "failed",
                    "returncode": int(rc),
                    "checkpoint": str(task["ckpt_path"]) if task["ckpt_path"].exists() else None,
                    "log_path": str(task["log_path"]),
                    "device": str(dev),
                }
                results.append(result)

                print(f"[DONE] class={task['cls']} on dev={dev}, rc={rc}")

                if rc != 0 and not args.continue_on_error:
                    for p2, f2, task2, dev2 in procs:
                        try:
                            p2.terminate()
                        except Exception:
                            pass
                    time.sleep(2)
                    for p2, f2, task2, dev2 in procs:
                        if p2.poll() is None:
                            try:
                                p2.kill()
                            except Exception:
                                pass
                        try:
                            f2.close()
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"class={task['cls']} 训练失败，日志见: {task['log_path']}"
                    )

    finally:
        for p, f, task, dev in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass

    return sorted(results, key=lambda x: x["class"])


def finalize_registry(tasks, results, args):
    result_map = {int(r["class"]): r for r in results}

    registry = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "save_root": str(Path(args.save_root).resolve()),
        "common": {
            "data_dir": str(args.data_dir),
            "sd_path": str(args.sd_path),
            "epoch": int(args.epoch),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "step_number": int(args.step_number),
            "cs_ratio": float(args.cs_ratio),
            "block_size": int(args.block_size),
            "classes": [int(c) for c in args.classes],
            "parallel": bool(args.parallel),
        },
        "classes": {},
    }

    for task in tasks:
        cls = int(task["cls"])
        entry = {
            "class": cls,
            "save_dir": str(task["save_dir"]),
            "log_path": str(task["log_path"]),
            "checkpoint": str(task["ckpt_path"]) if task["ckpt_path"].exists() else None,
            "suggested_port": int(args.suggested_base_port + task["idx"]),
            "status": result_map.get(cls, {}).get("status", "unknown"),
        }

        if task["ckpt_path"].exists() and args.export_phi:
            phi_dir = task["save_dir"] / "deploy_assets"
            phi_path, manifest_path = export_phi_from_checkpoint(task["ckpt_path"], phi_dir)
            entry["phi_path"] = str(phi_path)
            entry["phi_manifest"] = str(manifest_path)

        registry["classes"][str(cls)] = entry

    registry_path = Path(args.save_root) / "deploy_registry.json"
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[REGISTRY] 已生成: {registry_path}")
    return registry_path, registry


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--classes", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--data_dir", type=str, default="./data/ToolWear_RGB")
    parser.add_argument("--save_root", type=str, default="./checkpoints_4cls")
    parser.add_argument("--sd_path", type=str, default="./sd15")

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_number", type=int, default=8)
    parser.add_argument("--cs_ratio", type=float, default=0.1)
    parser.add_argument("--block_size", type=int, default=8)

    parser.add_argument("--hf_endpoint", type=str, default="")
    parser.add_argument("--visible_devices", type=str, default=None, help="串行模式下使用，如: 0 或 0,1")
    parser.add_argument("--parallel", type=str2bool, default=False)
    parser.add_argument("--devices", nargs="*", default=None, help="并行模式下每个训练进程绑定一个设备，如: 0 1 2 3")

    parser.add_argument("--skip_existing", type=str2bool, default=False)
    parser.add_argument("--continue_on_error", type=str2bool, default=False)
    parser.add_argument("--export_phi", type=str2bool, default=True)
    parser.add_argument("--suggested_base_port", type=int, default=19001)

    args = parser.parse_args()

    tasks = prepare_tasks(args)

    t0 = time.time()
    if args.parallel:
        results = run_parallel(tasks, args)
    else:
        results = run_serial(tasks, args)

    registry_path, registry = finalize_registry(tasks, results, args)

    failed = [r for r in results if r["status"] == "failed"]
    cost = time.time() - t0

    print("\n" + "=" * 80)
    print("训练总览")
    print("=" * 80)
    for cls in args.classes:
        entry = registry["classes"].get(str(cls), {})
        print(
            f"class={cls} | status={entry.get('status')} | "
            f"ckpt={entry.get('checkpoint')} | phi={entry.get('phi_path', None)} | "
            f"port={entry.get('suggested_port')}"
        )
    print(f"总耗时: {cost / 3600.0:.2f} 小时")
    print(f"部署注册表: {registry_path}")

    if failed:
        print("\n存在失败任务，请检查对应 log_path。")
        sys.exit(1)


if __name__ == "__main__":
    main()