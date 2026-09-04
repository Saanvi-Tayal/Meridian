"""Offline Road Network Graph & Spatial Geometric Snapper

Represents digital road networks as directed polyline segments with spatial indexing
(grid-based spatial hashing) for sub-millisecond candidate lookup and point-to-road projection.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import heapq


@dataclass
class RoadCandidate:
    """A candidate projection point on a road segment."""
    segment_id: str
    projected_point: np.ndarray  # [East, North]
    distance_m: float           # Perpendicular distance from query to road
    fraction_s: float           # Normalized progress along segment in [0.0, 1.0]
    road_heading_rad: float     # Heading of the road at the projected point


class RoadSegment:
    """Represents a directed road link defined by a 2D polyline."""

    def __init__(
        self,
        segment_id: str,
        start_node: str,
        end_node: str,
        polyline: np.ndarray,
        speed_limit_kmh: float = 80.0,
        is_one_way: bool = True,
    ):
        self.segment_id = segment_id
        self.start_node = start_node
        self.end_node = end_node
        self.polyline = np.array(polyline, dtype=float)
        self.speed_limit_kmh = speed_limit_kmh
        self.is_one_way = is_one_way

        # Precompute segment geometry & lengths
        diffs = np.diff(self.polyline, axis=0)
        self.sub_lengths = np.linalg.norm(diffs, axis=1)
        # Avoid division by zero on duplicate nodes
        self.sub_lengths = np.maximum(1e-6, self.sub_lengths)
        self.cum_lengths = np.pad(np.cumsum(self.sub_lengths), (1, 0), mode="constant")
        self.total_length_m = float(self.cum_lengths[-1])

        # Sub-segment headings: clockwise from North (Y-axis)
        # Heading psi: East is +X, North is +Y -> psi = atan2(dEast, dNorth)
        self.sub_headings = np.arctan2(diffs[:, 0], diffs[:, 1])

        # Bounding box
        self.min_bounds = np.min(self.polyline, axis=0)
        self.max_bounds = np.max(self.polyline, axis=0)

    def project_point(self, query_pt: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Projects a 2D point [East, North] onto this polyline segment.
        Returns:
            (closest_point, distance_m, fraction_s, heading_at_proj)
        """
        best_dist = float("inf")
        best_proj = self.polyline[0].copy()
        best_dist_along = 0.0
        best_heading = self.sub_headings[0]

        q = np.array(query_pt, dtype=float)
        num_sub = len(self.sub_lengths)

        for i in range(num_sub):
            p1 = self.polyline[i]
            p2 = self.polyline[i + 1]
            seg_vec = p2 - p1
            seg_len_sq = float(np.dot(seg_vec, seg_vec))

            if seg_len_sq < 1e-10:
                proj = p1
                t = 0.0
            else:
                t = np.clip(np.dot(q - p1, seg_vec) / seg_len_sq, 0.0, 1.0)
                proj = p1 + t * seg_vec

            d = float(np.linalg.norm(q - proj))
            if d < best_dist:
                best_dist = d
                best_proj = proj
                best_dist_along = self.cum_lengths[i] + t * self.sub_lengths[i]
                best_heading = float(self.sub_headings[i])

        fraction_s = float(np.clip(best_dist_along / max(1e-3, self.total_length_m), 0.0, 1.0))
        return best_proj, best_dist, fraction_s, best_heading


class RoadNetwork:
    """
    Offline topological road network graph with spatial grid indexing.
    """

    def __init__(self, cell_size: float = 50.0):
        self.cell_size = cell_size
        self.segments: Dict[str, RoadSegment] = {}
        self.adjacency: Dict[str, List[str]] = {}  # start_node -> list of segment_ids
        self.spatial_grid: Dict[Tuple[int, int], List[str]] = {}

    def add_segment(self, segment: RoadSegment):
        """Adds a road segment to the graph and spatial index."""
        seg_id = segment.segment_id
        self.segments[seg_id] = segment

        if segment.start_node not in self.adjacency:
            self.adjacency[segment.start_node] = []
        self.adjacency[segment.start_node].append(seg_id)

        # Index into spatial grid cells covered by the bounding box (with margin)
        min_cell_x = int(np.floor((segment.min_bounds[0] - 10.0) / self.cell_size))
        max_cell_x = int(np.floor((segment.max_bounds[0] + 10.0) / self.cell_size))
        min_cell_y = int(np.floor((segment.min_bounds[1] - 10.0) / self.cell_size))
        max_cell_y = int(np.floor((segment.max_bounds[1] + 10.0) / self.cell_size))

        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                cell_key = (cx, cy)
                if cell_key not in self.spatial_grid:
                    self.spatial_grid[cell_key] = []
                self.spatial_grid[cell_key].append(seg_id)

    def find_candidates(
        self,
        query_pt: np.ndarray,
        search_radius_m: float = 30.0,
        max_candidates: int = 5
    ) -> List[RoadCandidate]:
        """
        Retrieves all road segments within search_radius_m of query_pt,
        returning projection candidates sorted by distance.
        """
        cx = int(np.floor(query_pt[0] / self.cell_size))
        cy = int(np.floor(query_pt[1] / self.cell_size))

        # Check neighborhood cells
        radius_cells = int(np.ceil(search_radius_m / self.cell_size))
        considered_segments = set()

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cell = (cx + dx, cy + dy)
                if cell in self.spatial_grid:
                    for s_id in self.spatial_grid[cell]:
                        considered_segments.add(s_id)

        # If spatial grid was empty, fallback to searching all segments
        if not considered_segments:
            considered_segments = set(self.segments.keys())

        candidates: List[RoadCandidate] = []
        for s_id in considered_segments:
            seg = self.segments[s_id]
            proj, dist, frac_s, heading = seg.project_point(query_pt)
            if dist <= search_radius_m:
                candidates.append(
                    RoadCandidate(
                        segment_id=s_id,
                        projected_point=proj,
                        distance_m=dist,
                        fraction_s=frac_s,
                        road_heading_rad=heading
                    )
                )

        # Sort candidates by distance ascending
        candidates.sort(key=lambda c: c.distance_m)
        return candidates[:max_candidates]

    def shortest_network_distance(
        self,
        from_seg_id: str,
        from_s: float,
        to_seg_id: str,
        to_s: float,
        max_depth_m: float = 200.0
    ) -> float:
        """
        Calculates shortest drivable path distance along the road network
        between two candidate positions.
        """
        from_seg = self.segments.get(from_seg_id)
        to_seg = self.segments.get(to_seg_id)

        if from_seg is None or to_seg is None:
            return float("inf")

        # Case 1: Same segment
        if from_seg_id == to_seg_id:
            if to_s >= from_s:
                return (to_s - from_s) * from_seg.total_length_m
            elif not from_seg.is_one_way:
                return (from_s - to_s) * from_seg.total_length_m
            # Reversing on a one-way segment is prohibited / heavily penalized
            return float("inf")

        # Case 2: Graph search (Dijkstra) from from_seg.end_node
        dist_to_end = (1.0 - from_s) * from_seg.total_length_m
        dist_from_start = to_s * to_seg.total_length_m

        # Priority queue entries: (cost_so_far, current_node)
        pq = [(dist_to_end, from_seg.end_node)]
        visited: Dict[str, float] = {}

        while pq:
            cost, u = heapq.heappop(pq)
            if cost > max_depth_m:
                break
            if u in visited and visited[u] <= cost:
                continue
            visited[u] = cost

            # Check if to_seg originates from node u
            if u == to_seg.start_node:
                return cost + dist_from_start

            # Explore outgoing segments
            for next_seg_id in self.adjacency.get(u, []):
                next_seg = self.segments[next_seg_id]
                next_cost = cost + next_seg.total_length_m
                if next_seg.end_node not in visited or visited[next_seg.end_node] > next_cost:
                    heapq.heappush(pq, (next_cost, next_seg.end_node))

        # Disconnected or exceeding max search horizon
        return float("inf")

    @classmethod
    def from_trajectory(
        cls,
        east_coords: np.ndarray,
        north_coords: np.ndarray,
        segment_len_m: float = 100.0,
        cell_size: float = 50.0
    ) -> "RoadNetwork":
        """
        Constructs a digital road network corridor from a verified GNSS/CAN trajectory.
        Chops the trajectory into connected directed RoadSegments.
        """
        net = cls(cell_size=cell_size)
        N = len(east_coords)
        if N < 2:
            return net

        pts = np.column_stack([east_coords, north_coords])
        diffs = np.diff(pts, axis=0)
        lens = np.linalg.norm(diffs, axis=1)
        cum_dist = np.pad(np.cumsum(lens), (1, 0), mode="constant")

        start_idx = 0
        seg_counter = 0

        while start_idx < N - 1:
            target_dist = cum_dist[start_idx] + segment_len_m
            end_idx = np.searchsorted(cum_dist, target_dist)
            end_idx = min(N - 1, max(start_idx + 1, end_idx))

            sub_poly = pts[start_idx:end_idx + 1]
            seg_id = f"seg_{seg_counter:04d}"
            start_node = f"node_{seg_counter:04d}"
            end_node = f"node_{seg_counter + 1:04d}"

            segment = RoadSegment(
                segment_id=seg_id,
                start_node=start_node,
                end_node=end_node,
                polyline=sub_poly,
                is_one_way=True
            )
            net.add_segment(segment)

            seg_counter += 1
            start_idx = end_idx

        return net
