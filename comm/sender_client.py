from __future__ import annotations

import argparse
import asyncio
import numpy as np

from comm.protocol import (
    Message,
    read_message,
    write_message,
    array_to_npy_bytes,
    npy_bytes_to_array,
)


def fake_signal(n: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n, dtype=np.float32)

    x = (
        0.62 * np.sin(2 * np.pi * 7 * t)
        + 0.22 * np.sin(2 * np.pi * 19 * t + 0.6)
        + 0.12 * np.sin(2 * np.pi * 33 * t + 1.1)
    )
    x += 0.18 * np.exp(-((t - 0.33) / 0.02) ** 2)
    x -= 0.16 * np.exp(-((t - 0.74) / 0.03) ** 2)
    x += 0.03 * rng.standard_normal(n)
    x = x.astype(np.float32)
    return x


def make_measurement(signal: np.ndarray, m: int = 64, seed: int = 0) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal((m, len(signal))).astype(np.float32)
    phi = phi / (np.linalg.norm(phi, axis=1, keepdims=True) + 1e-6)
    y = phi @ signal
    return y.astype(np.float32)


class IDMClient:
    def __init__(
        self,
        host: str,
        port: int,
        device_id: str,
        secret: str,
        timeout: float = 15.0,
    ):
        self.host = host
        self.port = port
        self.device_id = device_id
        self.secret = secret
        self.timeout = timeout

        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.seq = 1

    async def connect(self) -> None:
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

        hello = Message(
            kind="hello",
            device_id=self.device_id,
            seq=0,
            meta={"role": "sender", "protocol": 1},
            payload=b"",
        )
        await write_message(self.writer, hello, self.secret)
        reply = await asyncio.wait_for(read_message(self.reader, self.secret), timeout=self.timeout)
        if reply.kind != "hello_ack":
            raise RuntimeError(f"unexpected handshake reply: {reply.kind}")

    async def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass

    async def send_heartbeat(self) -> dict:
        assert self.reader is not None and self.writer is not None

        msg = Message(
            kind="heartbeat",
            device_id=self.device_id,
            seq=self.seq,
            meta={},
            payload=b"",
        )
        self.seq += 1
        await write_message(self.writer, msg, self.secret)
        reply = await asyncio.wait_for(read_message(self.reader, self.secret), timeout=self.timeout)
        if reply.kind != "heartbeat_ack":
            raise RuntimeError(f"unexpected heartbeat reply: {reply.kind}")
        return reply.meta

    async def send_measurement(self, measurement: np.ndarray, meta: dict) -> dict:
        assert self.reader is not None and self.writer is not None

        seq = self.seq
        self.seq += 1

        payload = array_to_npy_bytes(np.asarray(measurement, dtype=np.float32))
        msg = Message(
            kind="data",
            device_id=self.device_id,
            seq=seq,
            meta=meta,
            payload=payload,
            payload_format="npy",
            codec="zlib",
        )
        await write_message(self.writer, msg, self.secret)

        ack = await asyncio.wait_for(read_message(self.reader, self.secret), timeout=self.timeout)
        if ack.kind != "ack" or ack.seq != seq:
            raise RuntimeError(f"unexpected ack: kind={ack.kind}, seq={ack.seq}, expected={seq}")

        result = await asyncio.wait_for(read_message(self.reader, self.secret), timeout=self.timeout)
        if result.kind == "error":
            raise RuntimeError(result.meta.get("message", "unknown error"))
        if result.kind != "result" or result.seq != seq:
            raise RuntimeError(f"unexpected result: kind={result.kind}, seq={result.seq}, expected={seq}")

        reconstruction = np.asarray(npy_bytes_to_array(result.payload), dtype=np.float32)

        return {
            "ack_meta": ack.meta,
            "result_meta": result.meta,
            "reconstruction": reconstruction,
        }


async def amain(args) -> None:
    client = IDMClient(
        host=args.host,
        port=args.port,
        device_id=args.device_id,
        secret=args.secret,
    )
    await client.connect()

    try:
        hb = await client.send_heartbeat()
        print("heartbeat:", hb)

        for i in range(args.num_windows):
            x = fake_signal(n=args.n, seed=args.seed + i)
            y = make_measurement(x, m=args.m, seed=args.seed + i)

            resp = await client.send_measurement(
                y,
                meta={
                    "sample_rate": args.sample_rate,
                    "original_length": len(x),
                    "phi_id": args.phi_id,
                    "window_index": i,
                },
            )

            print(f"[window {i}] ack={resp['ack_meta']}")
            print(f"[window {i}] result_meta={resp['result_meta']}")
            print(f"[window {i}] reconstruction_shape={resp['reconstruction'].shape}")
            print(f"[window {i}] reconstruction_head={resp['reconstruction'][:8]}")
            await asyncio.sleep(args.interval)

    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19001)
    parser.add_argument("--device-id", default="tx-001")
    parser.add_argument("--secret", default="demo-secret")
    parser.add_argument("--sample-rate", type=int, default=25600)
    parser.add_argument("--phi-id", default="phi_demo_v1")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval", type=float, default=0.3)
    args = parser.parse_args()

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()