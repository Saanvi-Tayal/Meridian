"""Unit Tests for Module 6: Offline Map-Matching & Road Network Constraint Engine"""

import unittest
import numpy as np

from src.map_matching.road_network import RoadNetwork, RoadSegment, RoadCandidate
from src.map_matching.hmm_matcher import HMMMapMatcher, MatchedState
from src.map_matching.curvature_feedback import RoadCurvatureConstraint
from src.fusion.engine import IntelligentDeadReckoningEngine


class TestMapMatching(unittest.TestCase):

    def setUp(self):
        # Create a simple road network: straight North road from (0, 0) to (0, 200)
        # and a connecting East road from (0, 200) to (200, 200)
        self.net = RoadNetwork(cell_size=50.0)

        # Segment 1: Heading North (heading = 0.0 rad)
        poly1 = np.array([
            [0.0, 0.0],
            [0.0, 100.0],
            [0.0, 200.0]
        ])
        self.seg1 = RoadSegment("seg_north", "node_0", "node_1", poly1, is_one_way=True)
        self.net.add_segment(self.seg1)

        # Segment 2: Heading East (heading = pi/2 rad)
        poly2 = np.array([
            [0.0, 200.0],
            [100.0, 200.0],
            [200.0, 200.0]
        ])
        self.seg2 = RoadSegment("seg_east", "node_1", "node_2", poly2, is_one_way=True)
        self.net.add_segment(self.seg2)

    def test_point_to_segment_projection(self):
        """Test orthogonal projection of points onto road segment."""
        # Query point at (5.0, 50.0) -> should project to (0.0, 50.0)
        q = np.array([5.0, 50.0])
        proj, dist, frac_s, heading = self.seg1.project_point(q)

        np.testing.assert_allclose(proj, [0.0, 50.0], atol=1e-3)
        self.assertAlmostEqual(dist, 5.0, places=3)
        self.assertAlmostEqual(frac_s, 50.0 / 200.0, places=3)
        self.assertAlmostEqual(heading, 0.0, places=3)

    def test_spatial_grid_lookup(self):
        """Test candidate retrieval via spatial hash grid."""
        # Query near (0, 25)
        cands = self.net.find_candidates(np.array([2.0, 25.0]), search_radius_m=10.0)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].segment_id, "seg_north")

        # Query near the intersection (0, 200)
        cands_inter = self.net.find_candidates(np.array([1.0, 199.0]), search_radius_m=15.0)
        seg_ids = [c.segment_id for c in cands_inter]
        self.assertIn("seg_north", seg_ids)
        self.assertIn("seg_east", seg_ids)

    def test_hmm_emission_probability(self):
        """Verify that smaller distance and aligned heading yield higher emission log-probability."""
        matcher = HMMMapMatcher(self.net, sigma_z=4.0)

        cand_close = RoadCandidate("seg_north", np.array([0.0, 50.0]), distance_m=1.0, fraction_s=0.25, road_heading_rad=0.0)
        cand_far = RoadCandidate("seg_north", np.array([0.0, 50.0]), distance_m=15.0, fraction_s=0.25, road_heading_rad=0.0)

        pos = np.array([1.0, 50.0])
        logp_close = matcher.emission_log_prob(cand_close, pos, vehicle_heading_rad=0.0)
        logp_far = matcher.emission_log_prob(cand_far, pos, vehicle_heading_rad=0.0)
        self.assertGreater(logp_close, logp_far)

        # Test heading penalty: driving against traffic (heading pi vs 0)
        logp_wrong_way = matcher.emission_log_prob(cand_close, pos, vehicle_heading_rad=np.pi)
        self.assertGreater(logp_close, logp_wrong_way)

    def test_hmm_viterbi_sequential_snapping(self):
        """Verify that a sequence of noisy coordinates is cleanly snapped to the road centerline."""
        matcher = HMMMapMatcher(self.net, sigma_z=4.0, beta=3.0)

        # Simulate noisy vehicle driving North with lateral drift: x ~ 3-5m, y = 10, 20, 30...
        y_true = np.linspace(10.0, 100.0, 10)
        x_noisy = np.array([3.0, 4.0, 2.5, 5.0, 3.5, 4.2, 3.8, 4.0, 4.5, 3.0])

        for t_idx, (x, y) in enumerate(zip(x_noisy, y_true)):
            raw_pos = np.array([x, y])
            state = matcher.step(
                time_s=float(t_idx),
                pred_pos=raw_pos,
                vehicle_heading_rad=0.0,
                delta_dist_dr=10.0
            )

            # Snapped East coordinate must be exactly on the road centerline (East = 0.0m)
            self.assertAlmostEqual(state.snapped_pos[0], 0.0, places=2)
            self.assertEqual(state.matched_segment_id, "seg_north")
            self.assertGreater(state.lateral_offset_m, 2.0)

    def test_road_curvature_constraint(self):
        """Test that curved segments provide correct yaw-rate constraints."""
        # Create an arc from 0 to 90 degrees turn
        angles = np.linspace(0, np.pi / 2, 10)
        radius = 100.0
        arc_x = radius * (1.0 - np.cos(angles))
        arc_y = radius * np.sin(angles)
        arc_poly = np.column_stack([arc_x, arc_y])

        curved_seg = RoadSegment("curved_seg", "c0", "c1", arc_poly)
        constraint = RoadCurvatureConstraint(trust_weight=0.5)

        # At speed 10 m/s on radius 100m, expected yaw rate is v / R = 10 / 100 = 0.1 rad/s
        kappa = constraint.calculate_segment_curvature(curved_seg, fraction_s=0.5)
        self.assertAlmostEqual(abs(kappa), 1.0 / radius, delta=0.015)

        # Measured gyro has zero rate (drifted) -> constraint should correct it towards 0.1 rad/s
        corr_yaw, exp_yaw = constraint.constrain_yaw_rate(
            measured_yaw_rate_rad_s=0.0,
            forward_speed_ms=10.0,
            segment=curved_seg,
            fraction_s=0.5
        )
        self.assertGreater(corr_yaw, 0.0)
        self.assertAlmostEqual(exp_yaw, 10.0 * kappa, places=3)

    def test_engine_map_matching_integration(self):
        """Test IntelligentDeadReckoningEngine integration with RoadNetwork."""
        engine = IntelligentDeadReckoningEngine(dt=0.1)
        engine.set_road_network(self.net, trust_curvature=True)

        engine.on_gnss_lost(time_s=0.0, p_entry=np.array([2.0, 10.0]), v_entry=10.0, heading_entry_rad=0.0)

        # Step dead reckoning
        pos, speed, heading = engine.step_dead_reckoning(
            yaw_rate_rad_s=0.0,
            features_window=None,
            v_rms_instant=0.5,
            is_stationary=False,
            time_s=0.1
        )

        # Check lane matched position
        snapped_pos = engine.get_lane_matched_position()
        self.assertIsNotNone(snapped_pos)
        # Snapped East coordinate should be on centerline (East = 0.0)
        self.assertAlmostEqual(snapped_pos[0], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
