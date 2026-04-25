from __future__ import annotations

import argparse
import asyncio
import logging
import shlex
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from comm.protocol import (
    Message,
    ProtocolError,
    IntegrityError,
    read_message,
    write_message,
    npy_bytes_to_array,
    array_to_npy_bytes,
)
from comm.adapters import (
    BaseIDMAdapter,
    MockIDMAdapter,
    ImportFunctionAdapter,
    SubprocessIDMAdapter,
    RealMyIDMAdapter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("idm-gateway")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


class IDMGatewayServer:
    def __init__(
        self,
        host: str,
        port: int,
        secret: str,
        adapter: BaseIDMAdapter,
        max_inflight: int = 2,
        dedupe_size: int = 4096,
    ):
        self.host = host
        self.port = port
        self.secret = secret
        self.adapter = adapter
        self.max_inflight = max_inflight
        self.dedupe_size = dedupe_size

        self.server: asyncio.AbstractServer | None = None
        self._infer_sem = asyncio.Semaphore(max_inflight)
        self._seen: OrderedDict[tuple[str, int], float] = OrderedDict()

    def _mark_seen(self, device_id: str, seq: int) -> bool:
        key = (device_id, seq)
        duplicate = key in self._seen
        self._seen[key] = time.time()
        while len(self._seen) > self.dedupe_size:
            self._seen.popitem(last=False)
        return duplicate

    async def start(self) -> None:
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        sockets = self.server.sockets or []
        addr_text = ", ".join(str(sock.getsockname()) for sock in sockets)
        log.info("gateway listening on %s", addr_text)

    async def serve_forever(self) -> None:
        await self.start()
        assert self.server is not None
        async with self.server:
            await self.server.serve_forever()

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            log.info("gateway stopped")

    async def _send(self, writer: asyncio.StreamWriter, lock: asyncio.Lock, msg: Message) -> None:
        async with lock:
            await write_message(writer, msg, self.secret)

    async def _process_data(
        self,
        msg: Message,
        measurement: np.ndarray,
        writer: asyncio.StreamWriter,
        lock: asyncio.Lock,
    ) -> None:
        try:
            async with self._infer_sem:
                result = await self.adapter.infer(measurement, msg.meta)

            reconstruction = np.asarray(result["reconstruction"], dtype=np.float32)
            meta = {k: v for k, v in result.items() if k != "reconstruction"}
            meta["origin_seq"] = msg.seq
            meta["status"] = meta.get("status", "ok")

            out = Message(
                kind="result",
                device_id="receiver-gateway",
                seq=msg.seq,
                meta=_json_safe(meta),
                payload=array_to_npy_bytes(reconstruction),
                payload_format="npy",
                codec="zlib",
            )
            await self._send(writer, lock, out)

        except Exception as exc:
            err = Message(
                kind="error",
                device_id="receiver-gateway",
                seq=msg.seq,
                meta={"origin_seq": msg.seq, "message": str(exc)},
                payload=b"",
                payload_format="bytes",
                codec="raw",
            )
            await self._send(writer, lock, err)

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        lock = asyncio.Lock()
        log.info("client connected: %s", peer)

        try:
            while True:
                msg = await read_message(reader, self.secret)

                if msg.kind == "hello":
                    reply = Message(
                        kind="hello_ack",
                        device_id="receiver-gateway",
                        seq=msg.seq,
                        meta={"status": "ok", "server": "idm-gateway", "protocol": 1},
                        payload=b"",
                    )
                    await self._send(writer, lock, reply)
                    continue

                if msg.kind == "heartbeat":
                    reply = Message(
                        kind="heartbeat_ack",
                        device_id="receiver-gateway",
                        seq=msg.seq,
                        meta={"status": "alive"},
                        payload=b"",
                    )
                    await self._send(writer, lock, reply)
                    continue

                if msg.kind != "data":
                    reply = Message(
                        kind="error",
                        device_id="receiver-gateway",
                        seq=msg.seq,
                        meta={"message": f"unsupported kind: {msg.kind}"},
                        payload=b"",
                    )
                    await self._send(writer, lock, reply)
                    continue

                if msg.payload_format != "npy":
                    reply = Message(
                        kind="error",
                        device_id="receiver-gateway",
                        seq=msg.seq,
                        meta={"origin_seq": msg.seq, "message": "data payload_format must be 'npy'"},
                        payload=b"",
                    )
                    await self._send(writer, lock, reply)
                    continue

                duplicate = self._mark_seen(msg.device_id, msg.seq)

                ack = Message(
                    kind="ack",
                    device_id="receiver-gateway",
                    seq=msg.seq,
                    meta={
                        "origin_seq": msg.seq,
                        "status": "duplicate" if duplicate else "accepted",
                    },
                    payload=b"",
                )
                await self._send(writer, lock, ack)

                if duplicate:
                    continue

                measurement = np.asarray(npy_bytes_to_array(msg.payload), dtype=np.float32).reshape(-1)
                asyncio.create_task(self._process_data(msg, measurement, writer, lock))

        except asyncio.IncompleteReadError:
            log.info("client disconnected: %s", peer)
        except (ProtocolError, IntegrityError) as exc:
            log.error("protocol error from %s: %s", peer, exc)
        except Exception:
            log.exception("unexpected server error for %s", peer)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


def build_adapter(args) -> BaseIDMAdapter:
    if args.adapter == "mock":
        return MockIDMAdapter()

    if args.adapter == "import":
        if not args.target:
            raise ValueError("import adapter requires --target module:function")
        return ImportFunctionAdapter(args.target)

    if args.adapter == "subprocess":
        if not args.target:
            raise ValueError("subprocess adapter requires --target 'python your_script.py'")
        return SubprocessIDMAdapter(shlex.split(args.target))

    if args.adapter == "myidm":
        if not args.checkpoint:
            raise ValueError("myidm adapter requires --checkpoint")
        return RealMyIDMAdapter(
            checkpoint=args.checkpoint,
            sd_path=args.sd_path,
            step_number=args.step_number,
            block_size=args.block_size,
            device=args.device,
            phi_path=args.phi_path,
            use_amp=args.use_amp,
        )

    raise ValueError(f"unknown adapter kind: {args.adapter}")
async def amain(args) -> None:
    adapter = build_adapter(args)
    server = IDMGatewayServer(
        host=args.host,
        port=args.port,
        secret=args.secret,
        adapter=adapter,
        max_inflight=args.max_inflight,
    )
    await server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19001)
    parser.add_argument("--secret", default="demo-secret")
    parser.add_argument("--adapter", choices=["mock", "import", "subprocess", "myidm"], default="myidm")
    parser.add_argument("--target", default=None)

    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--sd-path", default="./sd15")
    parser.add_argument("--step-number", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--phi-path", default=None)
    parser.add_argument("--device", default=None)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("1", "true", "yes", "y", "on")

    parser.add_argument("--use-amp", type=str2bool, default=None)

    # 非常重要：默认 1，避免 model.py 全局变量冲突
    parser.add_argument("--max-inflight", type=int, default=1)
    args = parser.parse_args()

    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()