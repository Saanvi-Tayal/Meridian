"""Road Curvature Constraint & Gyroscope Drift Bounding Engine

Exploits road geometry (curvature kappa(s)) along matched road links to provide
an expected yaw-rate constraint: omega_expected = v * kappa.
Prevents unconstrained gyroscope heading drift through long sweeping curved tunnels.
"""

from typing import Optional, Tuple
import numpy as np

from src.map_matching.road_network import RoadSegment


class RoadCurvatureConstraint:
    """
    Computes road curvature and derives kinematic constraints for heading integration.
    """

    def __init__(self, trust_weight: float = 0.35, max_yaw_rate_correction_rad_s: float = 0.05):
        """
        Args:
            trust_weight: Weight alpha in [0, 1] balancing gyro measurement vs road curvature constraint.
            max_yaw_rate_correction_rad_s: Maximum clamping limit on curvature feedback (rad/s).
        """
        self.trust_weight = trust_weight
        self.max_correction = max_yaw_rate_correction_rad_s

    @staticmethod
    def calculate_segment_curvature(segment: RoadSegment, fraction_s: float) -> float:
        """
        Estimates local curvature kappa (1/meters) at normalized position s on the segment.
        kappa > 0 indicates right turn, kappa < 0 indicates left turn.
        """
        num_sub = len(segment.sub_lengths)
        if num_sub < 2:
            return 0.0

        # Find which sub-segment we are currently on
        dist_along = fraction_s * segment.total_length_m
        idx = int(np.searchsorted(segment.cum_lengths, dist_along) - 1)
        idx = max(0, min(num_sub - 1, idx))

        # Look at heading change between adjacent sub-segments
        if idx < num_sub - 1:
            h1 = segment.sub_headings[idx]
            h2 = segment.sub_headings[idx + 1]
            dh = (h2 - h1 + np.pi) % (2.0 * np.pi) - np.pi
            ds = 0.5 * (segment.sub_lengths[idx] + segment.sub_lengths[idx + 1])
            return float(dh / max(1.0, ds))
        elif idx > 0:
            h1 = segment.sub_headings[idx - 1]
            h2 = segment.sub_headings[idx]
            dh = (h2 - h1 + np.pi) % (2.0 * np.pi) - np.pi
            ds = 0.5 * (segment.sub_lengths[idx - 1] + segment.sub_lengths[idx])
            return float(dh / max(1.0, ds))

        return 0.0

    def constrain_yaw_rate(
        self,
        measured_yaw_rate_rad_s: float,
        forward_speed_ms: float,
        segment: Optional[RoadSegment],
        fraction_s: float,
        is_high_confidence: bool = True
    ) -> Tuple[float, float]:
        """
        Bounds and corrects measured yaw rate using expected road curvature.
        Returns:
            (corrected_yaw_rate, expected_yaw_rate)
        """
        if segment is None or not is_high_confidence or forward_speed_ms < 1.0:
            return measured_yaw_rate_rad_s, 0.0

        kappa = self.calculate_segment_curvature(segment, fraction_s)
        expected_yaw_rate = forward_speed_ms * kappa

        # Innovation between measured and expected
        diff = expected_yaw_rate - measured_yaw_rate_rad_s
        correction = np.clip(self.trust_weight * diff, -self.max_correction, self.max_correction)
        corrected_yaw = float(measured_yaw_rate_rad_s + correction)

        return corrected_yaw, expected_yaw_rate
