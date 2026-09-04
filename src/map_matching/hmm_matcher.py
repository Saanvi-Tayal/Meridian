"""Hidden Markov Model (HMM) Map Matcher

Implements the Newson & Krumm HMM map matching formulation with online Viterbi decoding
for real-time (10Hz / 200Hz) lane-level vehicle road snapping.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.map_matching.road_network import RoadNetwork, RoadCandidate


@dataclass
class MatchedState:
    """Represents a map-matched vehicle position and metadata."""
    time_s: float
    snapped_pos: np.ndarray        # [East, North] in meters (lane-snapped)
    raw_pos: np.ndarray            # Unsnapped dead-reckoning position
    matched_segment_id: str        # Active road segment ID
    matched_heading_rad: float     # Road orientation at the snapped point
    fraction_s: float              # Progress along road segment [0, 1]
    lateral_offset_m: float        # Distance between raw dead-reckoning and snapped road
    log_likelihood: float          # Confidence log-probability


class HMMMapMatcher:
    """
    Real-time Hidden Markov Model Map Matcher.
    Binds noisy dead-reckoning coordinates to topological road links.
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        sigma_z: float = 4.0,
        beta: float = 3.0,
        search_radius_m: float = 35.0,
        max_candidates: int = 5,
    ):
        """
        Args:
            road_network: Offline topological road graph.
            sigma_z: Standard deviation of GPS/DR orthogonal noise (meters).
            beta: Exponential distribution scale parameter for distance discrepancy (meters).
            search_radius_m: Maximum search radius around query point.
            max_candidates: Maximum number of road candidates per step.
        """
        self.network = road_network
        self.sigma_z = sigma_z
        self.beta = beta
        self.search_radius_m = search_radius_m
        self.max_candidates = max_candidates

        # Viterbi state tracking
        # Each entry in trellis: Dict[candidate_key, log_prob]
        self.active_viterbi_scores: Dict[str, float] = {}
        self.active_candidates: Dict[str, RoadCandidate] = {}
        self.history: List[MatchedState] = []

    def emission_log_prob(
        self,
        candidate: RoadCandidate,
        pred_pos: np.ndarray,
        vehicle_heading_rad: float
    ) -> float:
        """
        Calculates log emission probability:
        log p(z_t | c_i) = -0.5*log(2*pi*sigma_z^2) - (d^2)/(2*sigma_z^2) + log(heading_penalty)
        """
        dist_sq = candidate.distance_m ** 2
        log_p_dist = -0.5 * np.log(2.0 * np.pi * (self.sigma_z ** 2)) - (dist_sq / (2.0 * (self.sigma_z ** 2)))

        # Heading alignment penalty: cos(psi_vehicle - psi_road)
        d_heading = candidate.road_heading_rad - vehicle_heading_rad
        cos_diff = float(np.cos(d_heading))
        # Penalize driving against traffic or perpendicular to road (min threshold 0.01 to avoid log(0))
        heading_weight = max(0.01, cos_diff)
        log_p_heading = np.log(heading_weight)

        return float(log_p_dist + log_p_heading)

    def transition_log_prob(
        self,
        from_cand: RoadCandidate,
        to_cand: RoadCandidate,
        delta_dist_dr: float
    ) -> float:
        """
        Calculates log transition probability:
        log p(c_{j, t} | c_{i, t-1}) = -log(beta) - (|d_network - delta_d_DR|) / beta
        """
        d_net = self.network.shortest_network_distance(
            from_seg_id=from_cand.segment_id,
            from_s=from_cand.fraction_s,
            to_seg_id=to_cand.segment_id,
            to_s=to_cand.fraction_s,
            max_depth_m=max(50.0, delta_dist_dr * 3.0)
        )

        if np.isinf(d_net):
            # Fallback to Euclidean candidate-to-candidate distance if network graph is disconnected
            d_net = float(np.linalg.norm(to_cand.projected_point - from_cand.projected_point))

        discrepancy = abs(d_net - delta_dist_dr)
        log_p_trans = -np.log(self.beta) - (discrepancy / self.beta)
        return float(log_p_trans)

    def step(
        self,
        time_s: float,
        pred_pos: np.ndarray,
        vehicle_heading_rad: float,
        delta_dist_dr: float
    ) -> MatchedState:
        """
        Executes one real-time HMM Viterbi update step for an incoming dead-reckoning coordinate.
        """
        candidates = self.network.find_candidates(
            query_pt=pred_pos,
            search_radius_m=self.search_radius_m,
            max_candidates=self.max_candidates
        )

        # Fallback: if no roads found within search radius, keep raw position
        if not candidates:
            state = MatchedState(
                time_s=time_s,
                snapped_pos=np.array(pred_pos, dtype=float),
                raw_pos=np.array(pred_pos, dtype=float),
                matched_segment_id="none",
                matched_heading_rad=vehicle_heading_rad,
                fraction_s=0.0,
                lateral_offset_m=0.0,
                log_likelihood=-100.0
            )
            self.history.append(state)
            return state

        current_scores: Dict[str, float] = {}
        current_cands: Dict[str, RoadCandidate] = {}

        # Case 1: Initial step
        if not self.active_viterbi_scores:
            for idx, cand in enumerate(candidates):
                c_key = f"{cand.segment_id}_{idx}"
                emit_logp = self.emission_log_prob(cand, pred_pos, vehicle_heading_rad)
                current_scores[c_key] = emit_logp
                current_cands[c_key] = cand
        else:
            # Case 2: Viterbi forward dynamic programming step
            for idx, cand in enumerate(candidates):
                c_key = f"{cand.segment_id}_{idx}"
                emit_logp = self.emission_log_prob(cand, pred_pos, vehicle_heading_rad)

                best_prev_score = -float("inf")
                for prev_key, prev_score in self.active_viterbi_scores.items():
                    prev_cand = self.active_candidates[prev_key]
                    trans_logp = self.transition_log_prob(prev_cand, cand, delta_dist_dr)
                    total_score = prev_score + trans_logp + emit_logp

                    if total_score > best_prev_score:
                        best_prev_score = total_score

                current_scores[c_key] = best_prev_score
                current_cands[c_key] = cand

        # Select highest scoring candidate
        best_key = max(current_scores, key=current_scores.get)
        best_cand = current_cands[best_key]
        best_logp = current_scores[best_key]

        # Update running trellis state
        self.active_viterbi_scores = current_scores
        self.active_candidates = current_cands

        matched = MatchedState(
            time_s=time_s,
            snapped_pos=best_cand.projected_point.copy(),
            raw_pos=np.array(pred_pos, dtype=float),
            matched_segment_id=best_cand.segment_id,
            matched_heading_rad=best_cand.road_heading_rad,
            fraction_s=best_cand.fraction_s,
            lateral_offset_m=best_cand.distance_m,
            log_likelihood=best_logp
        )
        self.history.append(matched)
        return matched

    def reset(self):
        """Resets Viterbi history and active trellis."""
        self.active_viterbi_scores.clear()
        self.active_candidates.clear()
        self.history.clear()
