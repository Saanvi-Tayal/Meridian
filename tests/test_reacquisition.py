"""Unit Tests for GNSS Reacquisition Adaptive Calibration & Error-Feedback Engine"""

import unittest
import numpy as np

from src.fusion.reacquisition import (
    GNSSReacquisitionManager,
    OutageSummary,
    CalibrationUpdate,
    wrap_angle_rad,
)
from src.fusion.engine import IntelligentDeadReckoningEngine


class TestGNSSReacquisition(unittest.TestCase):

    def setUp(self):
        self.mgr = GNSSReacquisitionManager(
            gyro_learning_rate=0.5,
            speed_learning_rate=0.5,
            smoothing_decay=2.0,
            min_outage_duration_s=2.0
        )

    def test_angle_wrapping(self):
        self.assertAlmostEqual(wrap_angle_rad(0.0), 0.0)
        self.assertAlmostEqual(wrap_angle_rad(np.pi), -np.pi)
        self.assertAlmostEqual(wrap_angle_rad(3 * np.pi), -np.pi)
        self.assertAlmostEqual(wrap_angle_rad(-3 * np.pi), -np.pi)
        self.assertAlmostEqual(wrap_angle_rad(np.pi / 2), np.pi / 2)

    def test_along_and_cross_track_error_decomposition(self):
        """
        Vehicle heading is due North (heading = 0 rad).
        Along-track is +North (+Y).
        Cross-track is +East (+X).
        """
        self.mgr.notify_gnss_lost(time_s=10.0, p_entry=np.array([0.0, 0.0]), heading_entry_rad=0.0)

        # Vehicle predicted exit at (0, 100), but true GNSS is at (10, 110)
        p_pred_exit = np.array([0.0, 100.0])
        p_gnss_exit = np.array([10.0, 110.0]) # +10m East (cross-track), +10m North (along-track)

        summary, calib = self.mgr.notify_gnss_restored(
            exit_time_s=20.0,
            p_pred_exit=p_pred_exit,
            v_pred_exit=10.0,
            heading_pred_exit_rad=0.0,
            distance_pred_m=100.0,
            p_gnss_exit=p_gnss_exit,
            v_gnss_exit=10.5,
            heading_gnss_exit_rad=0.05
        )

        self.assertAlmostEqual(summary.duration_s, 10.0)
        self.assertAlmostEqual(summary.along_track_error_m, 10.0, places=3)
        self.assertAlmostEqual(summary.cross_track_error_m, 10.0, places=3)
        self.assertAlmostEqual(summary.total_pos_error_m, np.sqrt(10**2 + 10**2), places=3)
        self.assertAlmostEqual(summary.heading_error_rad, 0.05, places=3)

    def test_adaptive_calibration_feedback(self):
        """
        Test that along-track undershoot increases speed scale factor,
        and positive heading error updates gyro bias.
        """
        self.mgr.notify_gnss_lost(time_s=50.0, p_entry=np.array([0.0, 0.0]), heading_entry_rad=0.0)

        # Distance predicted = 100m, but true GNSS traveled 110m (along-track error = +10m)
        p_pred_exit = np.array([0.0, 100.0])
        p_gnss_exit = np.array([0.0, 110.0])

        summary, calib = self.mgr.notify_gnss_restored(
            exit_time_s=60.0, # 10s duration
            p_pred_exit=p_pred_exit,
            v_pred_exit=10.0,
            heading_pred_exit_rad=0.0,
            distance_pred_m=100.0,
            p_gnss_exit=p_gnss_exit,
            v_gnss_exit=11.0,
            heading_gnss_exit_rad=0.10 # GNSS is +0.10 rad (~5.7 deg) turned
        )

        # speed_lr = 0.5, rel_err = 10 / 100 = 0.10 -> speed scale should increase by 0.5 * 0.10 = 0.05 (scale = 1.05)
        self.assertAlmostEqual(calib.new_speed_scale, 1.05, places=3)
        self.assertAlmostEqual(self.mgr.active_speed_scale, 1.05, places=3)

        # gyro_lr = 0.5, heading_err / duration = 0.10 / 10 = 0.01 rad/s -> delta gyro bias = -0.5 * 0.01 = -0.005 rad/s
        self.assertAlmostEqual(calib.new_gyro_bias_rad_s, -0.005, places=4)
        self.assertAlmostEqual(self.mgr.active_gyro_bias_rad_s, -0.005, places=4)

    def test_anti_teleport_visual_smoothing(self):
        """
        Verify that display position glides smoothly without instantaneous teleportation jump.
        """
        t_exit = 100.0
        p_pred_exit = np.array([50.0, 50.0])
        p_gnss_exit = np.array([55.0, 50.0]) # 5m jump East

        self.mgr.notify_gnss_lost(time_s=90.0, p_entry=np.array([0.0, 0.0]), heading_entry_rad=0.0)
        self.mgr.notify_gnss_restored(
            exit_time_s=t_exit,
            p_pred_exit=p_pred_exit,
            v_pred_exit=10.0,
            heading_pred_exit_rad=0.0,
            distance_pred_m=100.0,
            p_gnss_exit=p_gnss_exit,
            v_gnss_exit=10.0,
            heading_gnss_exit_rad=0.0
        )

        # Exactly at reacquisition time, display pos should match prediction (0 jump)
        p_disp_0 = self.mgr.get_smooth_display_pos(current_time_s=t_exit, current_gnss_pos=p_gnss_exit)
        np.testing.assert_allclose(p_disp_0, p_pred_exit, atol=1e-3)

        # At t = t_exit + 1.0s, offset has decayed by exp(-2.0 * 1.0) = exp(-2) ~ 0.135
        p_disp_1 = self.mgr.get_smooth_display_pos(current_time_s=t_exit + 1.0, current_gnss_pos=p_gnss_exit)
        expected_offset = np.array([5.0, 0.0]) * np.exp(-2.0)
        np.testing.assert_allclose(p_disp_1, p_gnss_exit - expected_offset, atol=1e-3)

        # After 3 seconds (> 4/decay), fully converged to GNSS
        p_disp_long = self.mgr.get_smooth_display_pos(current_time_s=t_exit + 3.0, current_gnss_pos=p_gnss_exit)
        np.testing.assert_allclose(p_disp_long, p_gnss_exit, atol=1e-3)

    def test_engine_closed_loop_adaptation(self):
        """
        Tests the IntelligentDeadReckoningEngine with sequential outages:
        Outage 1 creates feedback, which engine applies to Outage 2.
        """
        engine = IntelligentDeadReckoningEngine(dt=0.1)
        self.assertAlmostEqual(engine.reacquisition_mgr.active_speed_scale, 1.0)
        self.assertAlmostEqual(engine.reacquisition_mgr.active_gyro_bias_rad_s, 0.0)

        # Simulate Outage 1 (heading North, 10s)
        engine.on_gnss_lost(time_s=0.0, p_entry=np.array([0.0, 0.0]), v_entry=10.0, heading_entry_rad=0.0)
        for _ in range(100): # 10s at dt=0.1
            engine.step_dead_reckoning(
                yaw_rate_rad_s=0.01, # Unmodeled gyro bias
                features_window=None,
                v_rms_instant=0.5,
                is_stationary=False
            )

        # GNSS restored: GNSS heading was actually 0.0 rad (meaning dead reckoning drifted +0.1 rad)
        summary1, calib1 = engine.on_gnss_restored(
            time_s=10.0,
            p_gnss=np.array([2.0, 100.0]),
            v_gnss=10.0,
            heading_gnss_rad=0.0
        )

        # Engine should now have adapted positive gyro bias to cancel the positive yaw drift
        self.assertAlmostEqual(engine.reacquisition_mgr.active_gyro_bias_rad_s, 0.005, places=4)


if __name__ == "__main__":
    unittest.main()
