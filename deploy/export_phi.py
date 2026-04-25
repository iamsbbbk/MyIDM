# deploy/export_phi.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="./deploy_assets")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("checkpoint 格式异常，不是 dict")

    msd = ckpt.get("matrix_state_dict", None)
    if not isinstance(msd, dict) or "Phi" not in msd:
        raise RuntimeError("checkpoint 中没有 matrix_state_dict['Phi']")

    Phi = msd["Phi"].detach().float().cpu().numpy()
    cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phi_path = out_dir / "Phi.npy"
    manifest_path = out_dir / "manifest.json"

    np.save(phi_path, Phi)

    manifest = {
        "phi_shape": list(Phi.shape),
        "step_number": cfg.get("step_number", None),
        "block_size": cfg.get("block_size", None),
        "cs_ratio": cfg.get("cs_ratio", None),
        "target_class": cfg.get("target_class", None),
        "sd_path": cfg.get("sd_path", None),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("导出完成:")
    print(" -", phi_path)
    print(" -", manifest_path)


if __name__ == "__main__":
    main()