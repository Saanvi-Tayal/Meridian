"""
Integration test for the FastAPI Dead Reckoning Navigation Server.
"""
import unittest
from fastapi.testclient import TestClient
from src.server.app import app

class TestServerEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_and_static(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))

    def test_trips_listing(self):
        res = self.client.get("/api/trips")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("trips", data)
        self.assertGreater(data["count"], 0)

    def test_session_lifecycle_and_blackout(self):
        # 1. Get available trips
        trips_res = self.client.get("/api/trips")
        trip_id = trips_res.json()["trips"][0]

        # 2. Start session
        start_res = self.client.post("/api/session/start", json={"trip_id": trip_id})
        self.assertEqual(start_res.status_code, 200)
        self.assertEqual(start_res.json()["status"], "success")

        # 3. Step in GNSS-active mode
        step1_res = self.client.post("/api/engine/step", json={"step_size": 20})
        self.assertEqual(step1_res.status_code, 200)
        s1 = step1_res.json()
        self.assertFalse(s1["blackout_active"])
        self.assertIn("speed_kmh", s1)
        self.assertIn("current_position", s1)

        # 4. Toggle GPS Blackout
        toggle_res = self.client.post("/api/engine/toggle_blackout", json={"forced": True})
        self.assertEqual(toggle_res.status_code, 200)
        self.assertTrue(toggle_res.json()["blackout_active"])

        # 5. Step during Blackout (Dead Reckoning)
        step2_res = self.client.post("/api/engine/step", json={"step_size": 30})
        self.assertEqual(step2_res.status_code, 200)
        s2 = step2_res.json()
        self.assertTrue(s2["blackout_active"])
        self.assertIn("drift_percent", s2)
        self.assertGreaterEqual(s2["blackout_distance_m"], 0.0)

        # 6. Reset session
        reset_res = self.client.post("/api/engine/reset")
        self.assertEqual(reset_res.status_code, 200)
        self.assertEqual(reset_res.json()["frame"], 0)

if __name__ == '__main__':
    unittest.main()
