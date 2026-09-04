"""Unit tests for In-Vehicle Alignment and Vibration Pre-Filter modules."""

import unittest
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.calibration.aligner import InVehicleAligner
from src.filters.prefilter import VibrationPreFilter, StationaryDetector


class TestInVehicleAligner(unittest.TestCase):
    def setUp(self):
        self.aligner = InVehicleAligner(min_accel_dynamics_thresh=0.2)

    def test_vertical_axis_estimation_from_gravity(self):
        # Synthetic gravity pointing down in sensor frame (+Z)
        gravity = np.tile([0.0, 0.0, 9.80665], (100, 1))
        acc = gravity + np.random.normal(0, 0.05, (100, 3))
        
        u_z, g_norm = self.aligner.estimate_vertical_axis(acc, gravity_window=gravity)
        self.assertAlmostEqual(g_norm, 9.80665, places=2)
        np.testing.assert_allclose(u_z, [0.0, 0.0, 1.0], atol=1e-3)

    def test_calibration_and_transformation(self):
        # Phone mounted upside down along Y (Sensor Y points backwards = -Y_vehicle)
        N = 500
        t = np.linspace(0, 10, N)
        # True forward acceleration varies over time
        a_fwd = 1.0 + 0.8 * np.sin(2 * np.pi * 0.2 * t)
        speed = np.cumsum(a_fwd) * (10.0 / N)

        acc_s = np.zeros((N, 3))
        acc_s[:, 1] = -a_fwd # Measured along sensor -Y
        acc_s[:, 2] = 9.81   # Measured along sensor +Z
        gravity = np.tile([0.0, 0.0, 9.81], (N, 1))

        res = self.aligner.calibrate(acc=acc_s, gravity=gravity, speed_ref=speed)
        self.assertTrue(res.is_calibrated)

        # Forward axis (Y_v) transformation: sensor [-2.0] forward becomes vehicle [+2.0]
        v_sensor = np.array([0.0, -2.0, 9.81])
        v_veh = self.aligner.transform_vector(v_sensor)
        self.assertGreater(v_veh[1], 1.8)

    def test_mount_shift_detection(self):
        # Calibrate initial state
        gravity = np.tile([0.0, 0.0, 9.81], (50, 1))
        self.aligner.calibrate(acc=gravity, gravity=gravity)
        
        # Phone suddenly tilts 30 degrees (slip)
        tilt_angle = np.radians(30)
        tilted_gravity = np.tile([9.81 * np.sin(tilt_angle), 0, 9.81 * np.cos(tilt_angle)], (50, 1))
        shifted = self.aligner.detect_mount_shift(tilted_gravity)
        self.assertTrue(shifted)


class TestVibrationPreFilter(unittest.TestCase):
    def setUp(self):
        self.filter = VibrationPreFilter(sampling_rate_hz=10.0, cutoff_freq_hz=3.5)

    def test_pothole_shock_rejection(self):
        # Baseline acceleration with 1 sharp pothole spike
        N = 100
        acc = np.zeros((N, 3))
        acc[50, 2] = 28.0 # Sudden 28 m/s^2 shock spike
        
        cleaned, spike_mask = self.filter.reject_impulse_shocks(acc)
        self.assertTrue(spike_mask[50, 2])
        self.assertLess(cleaned[50, 2], 5.0)

    def test_stationary_detection(self):
        # Stationary signal (near zero variance)
        N = 50
        acc_stop = np.tile([0.0, 0.0, 9.81], (N, 1)) + np.random.normal(0, 0.01, (N, 3))
        gyro_stop = np.random.normal(0, 0.001, (N, 3))
        
        stops = self.filter.detect_stationary_states(acc_stop, gyro_stop)
        self.assertTrue(np.all(stops))


if __name__ == "__main__":
    unittest.main()
