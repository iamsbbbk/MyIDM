import unittest
import numpy as np

from comm.protocol import (
    Message,
    encode_message,
    decode_message_from_bytes,
    array_to_npy_bytes,
    npy_bytes_to_array,
    IntegrityError,
)


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.secret = "demo-secret"

    def test_roundtrip(self):
        arr = np.linspace(0, 1, 16, dtype=np.float32)

        msg = Message(
            kind="data",
            device_id="tx-001",
            seq=1,
            meta={"sample_rate": 25600},
            payload=array_to_npy_bytes(arr),
            payload_format="npy",
            codec="zlib",
        )

        blob = encode_message(msg, self.secret)
        out = decode_message_from_bytes(blob, self.secret)
        restored = npy_bytes_to_array(out.payload)

        self.assertEqual(out.kind, "data")
        self.assertEqual(out.device_id, "tx-001")
        self.assertEqual(out.seq, 1)
        np.testing.assert_allclose(restored, arr, atol=1e-6)

    def test_tamper_detected(self):
        arr = np.arange(8, dtype=np.float32)
        msg = Message(
            kind="data",
            device_id="tx-001",
            seq=2,
            meta={},
            payload=array_to_npy_bytes(arr),
            payload_format="npy",
            codec="zlib",
        )

        blob = bytearray(encode_message(msg, self.secret))
        blob[-1] ^= 0x01  # 篡改最后一个字节

        with self.assertRaises(IntegrityError):
            decode_message_from_bytes(bytes(blob), self.secret)


if __name__ == "__main__":
    unittest.main()