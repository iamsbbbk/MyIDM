from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class BaseIDMAdapter:
    async def infer(self, measurement: np.ndarray, meta: dict) -> dict:
        raise NotImplementedError


def _normalize_result(result: Any) -> dict:
    if isinstance(result, np.ndarray):
        return {
            "reconstruction": np.asarray(result, dtype=np.float32),
            "status": "ok",
        }

    if isinstance(result, dict):
        if "reconstruction" not in result:
            raise ValueError("adapter result must contain 'reconstruction'")
        out = dict(result)
        out["reconstruction"] = np.asarray(out["reconstruction"], dtype=np.float32)
        out.setdefault("status", "ok")
        return out

    raise TypeError("adapter result must be np.ndarray or dict")


class MockIDMAdapter(BaseIDMAdapter):
    """
    联调用的假 IDM：
    输入 measurement -> 插值 + 平滑 -> reconstruction
    """
    async def infer(self, measurement: np.ndarray, meta: dict) -> dict:
        await asyncio.sleep(0.05)

        m = np.asarray(measurement, dtype=np.float32).reshape(-1)
        out_len = int(meta.get("original_length", max(len(m) * 4, len(m))))

        x_old = np.arange(len(m), dtype=np.float32)
        x_new = np.linspace(0, len(m) - 1, out_len, dtype=np.float32)
        recon = np.interp(x_new, x_old, m).astype(np.float32)

        kernel = np.ones(9, dtype=np.float32) / 9.0
        recon = np.convolve(recon, kernel, mode="same").astype(np.float32)

        score = float(np.var(recon))
        label = "warning" if score > 0.08 else "normal"

        return {
            "reconstruction": recon,
            "score": score,
            "label": label,
            "status": "ok",
        }


class ImportFunctionAdapter(BaseIDMAdapter):
    """
    target 格式: "module.submodule:function_name"
    被调用函数签名建议:
        func(measurement: np.ndarray, meta: dict) -> np.ndarray | dict
    """
    def __init__(self, target: str):
        module_name, func_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        self.func = getattr(module, func_name)

    async def infer(self, measurement: np.ndarray, meta: dict) -> dict:
        if inspect.iscoroutinefunction(self.func):
            result = await self.func(measurement, meta)
        else:
            result = await asyncio.to_thread(self.func, measurement, meta)
        return _normalize_result(result)


class SubprocessIDMAdapter(BaseIDMAdapter):
    """
    如果你的真实 IDM 只有脚本入口，就用这个。
    约定子进程支持:
        python your_infer.py --input measurement.npy --meta meta.json --output result.npz

    result.npz 至少包含:
        reconstruction
    可选包含:
        score
        label
    """
    def __init__(self, command: list[str]):
        self.command = list(command)

    async def infer(self, measurement: np.ndarray, meta: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            input_path = tmp / "measurement.npy"
            meta_path = tmp / "meta.json"
            output_path = tmp / "result.npz"

            np.save(input_path, np.asarray(measurement, dtype=np.float32), allow_pickle=False)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

            proc = await asyncio.create_subprocess_exec(
                *self.command,
                "--input", str(input_path),
                "--meta", str(meta_path),
                "--output", str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"IDM subprocess failed, code={proc.returncode}, "
                    f"stderr={stderr.decode(errors='ignore')}"
                )

            with np.load(output_path, allow_pickle=False) as data:
                reconstruction = np.asarray(data["reconstruction"], dtype=np.float32)
                score = float(data["score"]) if "score" in data.files else float(np.var(reconstruction))
                if "label" in data.files:
                    label = str(data["label"].item())
                else:
                    label = "unknown"

            return {
                "reconstruction": reconstruction,
                "score": score,
                "label": label,
                "status": "ok",
                "stdout": stdout.decode(errors="ignore"),
            }

import asyncio

class RealMyIDMAdapter(BaseIDMAdapter):
    """
    直接包装你现在仓库里的 MyIDM 真实模型。
    不改 IDM 内核，只是把它变成一个可被 Gateway 调用的推理器。
    """
    def __init__(
        self,
        checkpoint: str,
        sd_path: str = "./sd15",
        step_number: int | None = None,
        block_size: int | None = None,
        device: str | None = None,
        phi_path: str | None = None,
        use_amp: bool | None = None,
    ):
        from deploy.idm_runtime import MyIDMRuntime

        self.runtime = MyIDMRuntime(
            checkpoint=checkpoint,
            sd_path=sd_path,
            step_number=step_number,
            block_size=block_size,
            device=device,
            phi_path=phi_path,
            use_amp=use_amp,
        )

    async def infer(self, measurement: np.ndarray, meta: dict) -> dict:
        # 注意：runtime 内部已经加锁，保证串行安全
        return await asyncio.to_thread(self.runtime.infer, measurement, meta)