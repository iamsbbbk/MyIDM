import base64
import hashlib
import json
import zlib
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict

import numpy as np
import torch


def _encode_ndarray(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    buf = BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return {
        "__ndarray__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
    }


def _decode_ndarray(obj: Dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(obj["data"].encode("utf-8"))
    buf = BytesIO(raw)
    arr = np.load(buf, allow_pickle=False)
    return arr


def _encode_bytes(b: bytes) -> Dict[str, Any]:
    return {
        "__bytes__": True,
        "data": base64.b64encode(b).decode("utf-8")
    }


def _decode_bytes(obj: Dict[str, Any]) -> bytes:
    return base64.b64decode(obj["data"].encode("utf-8"))


def _serialize_obj(obj: Any) -> Any:
    if torch.is_tensor(obj):
        obj = obj.detach().cpu().numpy()

    if isinstance(obj, np.ndarray):
        return _encode_ndarray(obj)

    if isinstance(obj, bytes):
        return _encode_bytes(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: _serialize_obj(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_serialize_obj(v) for v in obj]

    if isinstance(obj, tuple):
        return [_serialize_obj(v) for v in obj]

    return obj


def _deserialize_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__ndarray__", False):
            return _decode_ndarray(obj)
        if obj.get("__bytes__", False):
            return _decode_bytes(obj)
        return {k: _deserialize_obj(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_deserialize_obj(v) for v in obj]

    return obj


@dataclass
class SemanticPacket:
    header: Dict[str, Any] = field(default_factory=dict)
    semantic: Dict[str, Any] = field(default_factory=dict)
    anchors: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def payload_mode(self):
        return self.header.get("payload_mode", "M2")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header,
            "semantic": self.semantic,
            "anchors": self.anchors,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            header=d.get("header", {}),
            semantic=d.get("semantic", {}),
            anchors=d.get("anchors", {}),
            payload=d.get("payload", {}),
        )


def _compute_checksum(serialized_dict_without_checksum: Dict[str, Any]) -> str:
    raw = json.dumps(
        _serialize_obj(serialized_dict_without_checksum),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def packet_to_bytes(packet: SemanticPacket, compress: bool = True) -> bytes:
    packet_dict = packet.to_dict()
    packet_dict.setdefault("header", {})
    packet_dict["header"]["checksum"] = ""

    checksum = _compute_checksum(packet_dict)
    packet_dict["header"]["checksum"] = checksum

    raw = json.dumps(
        _serialize_obj(packet_dict),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    if compress:
        raw = zlib.compress(raw)

    return raw


def bytes_to_packet(packet_bytes: bytes, verify_checksum: bool = True) -> SemanticPacket:
    try:
        raw = zlib.decompress(packet_bytes)
    except Exception:
        raw = packet_bytes

    obj = json.loads(raw.decode("utf-8"))
    obj = _deserialize_obj(obj)

    if verify_checksum:
        header = obj.get("header", {})
        recv_checksum = header.get("checksum", "")
        obj["header"]["checksum"] = ""
        calc_checksum = _compute_checksum(obj)
        obj["header"]["checksum"] = recv_checksum

        if recv_checksum != calc_checksum:
            raise ValueError("SemanticPacket 校验失败: checksum 不一致")

    return SemanticPacket.from_dict(obj)