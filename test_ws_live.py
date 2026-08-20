"""Quick test for the WebSocket live detection endpoint."""
import requests
import json
import asyncio

async def test():
    # Login
    r = requests.post(
        "http://127.0.0.1:8000/auth/login",
        json={"email": "admin@fyp.com", "password": "Admin@123"}
    )
    token = r.json()["access_token"]
    print(f"Token obtained: {token[:30]}...")

    try:
        import websockets
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
        import websockets

    uri = f"ws://127.0.0.1:8000/model/live?token={token}"
    async with websockets.connect(uri) as ws:
        # 1) Should receive connection status
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"1) Connected: state={data['data']['state']}, sensors={data['data']['sensors']}")

        # 2) Start streaming with fast speed
        await ws.send(json.dumps({"action": "start", "speed": 0.1}))
        msg2 = await ws.recv()
        d2 = json.loads(msg2)
        print(f"2) Start: state={d2['data']['state']}")

        # 3) Get first 3 predictions
        for i in range(3):
            msg3 = await ws.recv()
            pred = json.loads(msg3)
            d = pred["data"]
            print(f"3.{i+1}) Row {d['row']}: attack={d['is_attack']}, prob={d['system_probability']:.4f}, anomalies={d['anomaly_count']}, actual={d['actual_label']}")

        # 4) Stop
        await ws.send(json.dumps({"action": "stop"}))
        msg4 = await ws.recv()
        d4 = json.loads(msg4)
        print(f"4) Stopped: {d4['data']['state']}, processed={d4['data'].get('total_processed', '?')}")

        print("\n✅ WebSocket live detection test PASSED!")

asyncio.run(test())
