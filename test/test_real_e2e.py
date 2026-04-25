# tests/test_real_e2e.py
from __future__ import annotations

import argparse
import asyncio
import math

import numpy as np
import torch

from comm.receiver_server import IDMGatewayServer
from comm.sender_client import IDMClient
from comm.adapters import RealMyIDMAdapter
from deploy.runtime_utils import make_A_AT_for_shape


def psnr01(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def make_demo_sample(height=32, width=32, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = xx / max(width - 1, 1)
    yy = yy / max(height - 1, 1)

    z = (
        0.35 * np.sin(2 * np.pi * 4 * xx)
        + 0.22 * np.cos(2 * np.pi * 3 * yy)
        + 0.18 * np.sin(2 * np.pi * (xx + yy) * 2.5)
    )
    z += 0.30 * np.exp(-((xx - 0.25) ** 2 + (yy - 0.68) ** 2) / 0.018)
    z -= 0.24 * np.exp(-((xx - 0.72) ** 2 + (yy - 0.32) ** 2) / 0.022)
    z += 0.03 * rng.standard_normal((height, width)).astype(np.float32)

    z = (z - z.min()) / (z.max() - z.min() + 1e-8)
    return z.astype(np.float32)


def load_phi_from_checkpoint(checkpoint: str) -> torch.Tensor:
    ckpt = torch.load(checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("checkpoint 格式错误")
    msd = ckpt.get("matrix_state_dict", {})
    if "Phi" not in msd:
        raise RuntimeError("checkpoint 中未找到 matrix_state_dict['Phi']")
    return msd["Phi"].detach().float().cpu()


def compress_with_phi(sample: np.ndarray, Phi: torch.Tensor, block_size: int) -> np.ndarray:
    """
    sample:
      [H, W] 或 [C, H, W]
    返回:
      y: [Bflat, q, L]
    """
    if sample.ndim == 2:
        x = torch.from_numpy(sample).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        channels = 1
    elif sample.ndim == 3:
        x = torch.from_numpy(sample).float().unsqueeze(1)  # [C,1,H,W]
        channels = sample.shape[0]
    else:
        raise ValueError(f"sample.ndim 必须是 2 或 3，当前为 {sample.ndim}")

    _, _, h, w = x.shape
    A_func, _ = make_A_AT_for_shape(h, w, block_size, Phi)

    with torch.no_grad():
        y = A_func(x).cpu().numpy().astype(np.float32)

    return y, channels, h, w


async def amain(args):
    adapter = RealMyIDMAdapter(
        checkpoint=args.checkpoint,
        sd_path=args.sd_path,
        step_number=args.step_number,
        block_size=args.block_size,
        device=args.device,
        phi_path=args.phi_path,
        use_amp=args.use_amp,
    )

    server = IDMGatewayServer(
        host=args.host,
        port=args.port,
        secret=args.secret,
        adapter=adapter,
        max_inflight=1,   # 强制串行，保护 model.py 中的全局状态
    )
    await server.start()

    client = IDMClient(
        host=args.host,
        port=args.port,
        device_id=args.device_id,
        secret=args.secret,
        timeout=args.timeout,
    )
    await client.connect()

    try:
        hb = await client.send_heartbeat()
        print("heartbeat:", hb)

        sample = make_demo_sample(args.height, args.width, seed=args.seed)
        Phi = load_phi_from_checkpoint(args.checkpoint)
        y, channels, h, w = compress_with_phi(sample, Phi, args.block_size)

        resp = await client.send_measurement(
            y,
            meta={
                "height": h,
                "width": w,
                "channels": channels,
                "block_size": args.block_size,
                "phi_source": "checkpoint",
                "window_index": 0,
                "sample_kind": "demo_2d",
            },
        )

        rec = resp["reconstruction"]
        # reconstruction 可能是 [1,H,W] 或 [H,W]
        if rec.ndim == 3 and rec.shape[0] == 1:
            rec = rec[0]
        elif rec.ndim == 2:
            pass
        else:
            raise RuntimeError(f"unexpected reconstruction shape: {rec.shape}")

        rec = np.clip(rec, 0.0, 1.0)
        p = psnr01(rec, sample)

        print("ack_meta:", resp["ack_meta"])
        print("result_meta:", resp["result_meta"])
        print("sample shape:", sample.shape)
        print("measurement shape:", y.shape)
        print("reconstruction shape:", rec.shape)
        print(f"PSNR={p:.2f} dB")

    finally:
        await client.close()
        await server.stop()


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19001)
    parser.add_argument("--secret", default="demo-secret")
    parser.add_argument("--device-id", default="tx-001")
    parser.add_argument("--timeout", type=float, default=60.0)

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sd-path", default="./sd15")
    parser.add_argument("--step-number", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--phi-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-amp", type=str2bool, default=None)

    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()