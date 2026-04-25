from __future__ import annotations

import asyncio
import io
import json
import struct
import time
import zlib
import hmac
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

MAGIC = b"IDM1"
HEADER_STRUCT = struct.Struct("!4sI")  # magic + header_len
MAX_HEADER = 64 * 1024
MAX_PAYLOAD = 64 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


class IntegrityError(ProtocolError):
    pass


@dataclass
class Message:
    kind: str
    device_id: str
    seq: int
    meta: Dict[str, Any] = field(default_factory=dict)
    payload: bytes = b""
    payload_format: str = "bytes"   # bytes | json | npy
    codec: str = "raw"              # raw | zlib
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))


def _secret_bytes(secret: str | bytes) -> bytes:
    return secret.encode("utf-8") if isinstance(secret, str) else secret


def array_to_npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


def npy_bytes_to_array(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)


def json_to_bytes(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def bytes_to_json(data: bytes) -> Dict[str, Any]:
    return json.loads(data.decode("utf-8"))


def _canonical_header_bytes(header_wo_sig: Dict[str, Any]) -> bytes:
    return json.dumps(
        header_wo_sig,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    ).encode("utf-8")


def encode_message(msg: Message, secret: str | bytes) -> bytes:
    secret_b = _secret_bytes(secret)

    if msg.codec == "zlib":
        payload_tx = zlib.compress(msg.payload, level=6)
    elif msg.codec == "raw":
        payload_tx = msg.payload
    else:
        raise ProtocolError(f"unsupported codec: {msg.codec}")

    header_wo_sig = {
        "version": 1,
        "kind": msg.kind,
        "device_id": msg.device_id,
        "seq": msg.seq,
        "ts_ms": msg.ts_ms,
        "payload_len": len(payload_tx),
        "payload_format": msg.payload_format,
        "codec": msg.codec,
        "meta": msg.meta,
        "crc32": zlib.crc32(payload_tx) & 0xFFFFFFFF,
    }

    sig = hmac.digest(
        secret_b,
        _canonical_header_bytes(header_wo_sig) + payload_tx,
        "sha256",
    ).hex()

    header = dict(header_wo_sig)
    header["sig"] = sig

    header_bytes = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    ).encode("utf-8")

    if len(header_bytes) > MAX_HEADER:
        raise ProtocolError(f"header too large: {len(header_bytes)}")
    if len(payload_tx) > MAX_PAYLOAD:
        raise ProtocolError(f"payload too large: {len(payload_tx)}")

    return HEADER_STRUCT.pack(MAGIC, len(header_bytes)) + header_bytes + payload_tx


def decode_message_from_bytes(blob: bytes, secret: str | bytes) -> Message:
    secret_b = _secret_bytes(secret)

    if len(blob) < HEADER_STRUCT.size:
        raise ProtocolError("blob too short")

    magic, header_len = HEADER_STRUCT.unpack(blob[:HEADER_STRUCT.size])
    if magic != MAGIC:
        raise ProtocolError("bad magic")
    if header_len > MAX_HEADER:
        raise ProtocolError("header too large")

    if len(blob) < HEADER_STRUCT.size + header_len:
        raise ProtocolError("incomplete header")

    header_bytes = blob[HEADER_STRUCT.size:HEADER_STRUCT.size + header_len]
    header = json.loads(header_bytes.decode("utf-8"))

    payload_tx = blob[HEADER_STRUCT.size + header_len:]
    if len(payload_tx) != header["payload_len"]:
        raise ProtocolError("payload length mismatch")

    if (zlib.crc32(payload_tx) & 0xFFFFFFFF) != header["crc32"]:
        raise IntegrityError("crc32 mismatch")

    recv_sig = header["sig"]
    header_wo_sig = dict(header)
    header_wo_sig.pop("sig", None)

    expected_sig = hmac.digest(
        secret_b,
        _canonical_header_bytes(header_wo_sig) + payload_tx,
        "sha256",
    ).hex()

    if not hmac.compare_digest(recv_sig, expected_sig):
        raise IntegrityError("hmac mismatch")

    codec = header["codec"]
    if codec == "zlib":
        payload = zlib.decompress(payload_tx)
    elif codec == "raw":
        payload = payload_tx
    else:
        raise ProtocolError(f"unsupported codec: {codec}")

    total_len = HEADER_STRUCT.size + header_len + header["payload_len"]
    if len(blob) != total_len:
        raise ProtocolError("trailing bytes detected")

    return Message(
        kind=header["kind"],
        device_id=header["device_id"],
        seq=header["seq"],
        meta=header.get("meta", {}),
        payload=payload,
        payload_format=header["payload_format"],
        codec=header["codec"],
        ts_ms=header["ts_ms"],
    )


async def write_message(writer: asyncio.StreamWriter, msg: Message, secret: str | bytes) -> None:
    blob = encode_message(msg, secret)
    writer.write(blob)
    await writer.drain()


async def read_message(reader: asyncio.StreamReader, secret: str | bytes) -> Message:
    prefix = await reader.readexactly(HEADER_STRUCT.size)
    magic, header_len = HEADER_STRUCT.unpack(prefix)

    if magic != MAGIC:
        raise ProtocolError("bad magic")
    if header_len > MAX_HEADER:
        raise ProtocolError("header too large")

    header_bytes = await reader.readexactly(header_len)
    header = json.loads(header_bytes.decode("utf-8"))

    payload_len = int(header["payload_len"])
    if payload_len > MAX_PAYLOAD:
        raise ProtocolError("payload too large")

    payload_tx = await reader.readexactly(payload_len)
    blob = prefix + header_bytes + payload_tx
    return decode_message_from_bytes(blob, secret)