import unittest

from comm.adapters import MockIDMAdapter
from comm.receiver_server import IDMGatewayServer
from comm.sender_client import IDMClient, fake_signal, make_measurement


class TestE2E(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.secret = "demo-secret"
        self.host = "127.0.0.1"
        self.port = 19001

        self.server = IDMGatewayServer(
            host=self.host,
            port=self.port,
            secret=self.secret,
            adapter=MockIDMAdapter(),
            max_inflight=2,
        )
        await self.server.start()

        self.client = IDMClient(
            host=self.host,
            port=self.port,
            device_id="tx-001",
            secret=self.secret,
        )
        await self.client.connect()

    async def asyncTearDown(self):
        await self.client.close()
        await self.server.stop()

    async def test_heartbeat(self):
        hb = await self.client.send_heartbeat()
        self.assertEqual(hb["status"], "alive")

    async def test_send_one_window(self):
        x = fake_signal(n=256, seed=1)
        y = make_measurement(x, m=64, seed=1)

        resp = await self.client.send_measurement(
            y,
            meta={
                "sample_rate": 25600,
                "original_length": len(x),
                "phi_id": "phi_demo_v1",
                "window_index": 0,
            },
        )

        self.assertEqual(resp["ack_meta"]["status"], "accepted")
        self.assertEqual(resp["result_meta"]["status"], "ok")
        self.assertEqual(resp["result_meta"]["origin_seq"], 2)  # heartbeat 用掉 seq=1
        self.assertEqual(resp["reconstruction"].shape[0], len(x))


if __name__ == "__main__":
    unittest.main()