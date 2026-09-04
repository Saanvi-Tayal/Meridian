"""Unit tests for the 15-state Error-State Kalman Filter and kinematic constraints."""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fusion.eskf import ErrorStateKalmanFilter
from src.fusion.constraints import NonHolonomicConstraint, ZeroVelocityUpdate, AISpeedConstraint


class TestESKF(unittest.TestCase):
    def setUp(self):
        self.eskf = ErrorStateKalmanFilter(
            init_pos=np.zeros(3),
            init_vel=np.array([0.0, 10.0, 0.0]), # Moving forward at 10 m/s
            init_quat=np.array([1.0, 0.0, 0.0, 0.0]) # Identity orientation
        )

    def test_predict_step(self):
        # Accelerating at +1 m/s^2 forward (Body Y)
        # Remember: Accelerometer at rest measures reaction to gravity [0, 0, +9.80665]
        a_meas = np.array([0.0, 1.0, 9.80665])
        w_meas = np.zeros(3)
        dt = 0.1

        init_pos_y = self.eskf.p_n[1]
        self.eskf.predict(a_meas, w_meas, dt)

        # Position should increase forward
        self.assertGreater(self.eskf.p_n[1], init_pos_y)
        # Velocity should increase
        self.assertGreater(self.eskf.v_n[1], 10.0)

    def test_gnss_update(self):
        # Provide GNSS fix
        pos_meas = np.array([5.0, 10.0, 0.0])
        vel_meas = np.array([0.0, 10.5, 0.0])
        
        cov_before = np.trace(self.eskf.P[0:3, 0:3])
        self.eskf.update_gnss(pos_meas, vel_meas, pos_std=1.0, vel_std=0.2)
        cov_after = np.trace(self.eskf.P[0:3, 0:3])

        # Covariance should decrease after measurement update
        self.assertLess(cov_after, cov_before)
        # Position should update by weighted gain K=0.5 towards measurement [5.0, 10.0]
        np.testing.assert_allclose(self.eskf.p_n[0:2], [2.5, 5.0], atol=0.2)

    def test_zupt_update(self):
        # Set non-zero velocity
        self.eskf.v_n = np.array([1.2, -0.8, 0.5])
        zupt = ZeroVelocityUpdate(sigma_zupt=0.01)
        y, H, R = zupt.get_measurement_model(self.eskf.v_n)
        self.eskf.apply_kalman_update(y, H, R)

        # Velocity should be pulled close to 0 m/s
        np.testing.assert_allclose(self.eskf.v_n, np.zeros(3), atol=0.05)

    def test_nhc_update(self):
        # Set lateral drifting velocity
        self.eskf.v_n = np.array([3.0, 15.0, 2.0]) # 3 m/s lateral slip, 2 m/s vertical
        nhc = NonHolonomicConstraint(sigma_lateral=0.05, sigma_vertical=0.05)
        y, H, R = nhc.get_measurement_model(self.eskf.R_nb, self.eskf.v_n)
        self.eskf.apply_kalman_update(y, H, R)

        # Body frame lateral & vertical velocity should be pulled near 0
        v_b = self.eskf.R_nb.T @ self.eskf.v_n
        self.assertLess(abs(v_b[0]), 0.5)
        self.assertLess(abs(v_b[2]), 0.5)

    def test_ai_speed_update(self):
        self.eskf.v_n = np.array([0.0, 12.0, 0.0]) # currently estimated at 12 m/s
        ai_speed = AISpeedConstraint(sigma_speed=0.5)
        y, H, R = ai_speed.get_measurement_model(self.eskf.R_nb, self.eskf.v_n, v_ai=20.0) # AI predicts 20 m/s
        self.eskf.apply_kalman_update(y, H, R)

        # Forward velocity should increase towards 20 m/s
        self.assertGreater(self.eskf.v_n[1], 15.0)


if __name__ == "__main__":
    unittest.main()
