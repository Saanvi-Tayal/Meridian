"""
Session Manager for Dead Reckoning Web Application.
Connects IO-VNBD dataset, Pre-filter, Alignment, SpeedNet, ES-EKF,
and HMM Map Matcher with interactive GPS Blackout control.
"""
from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.dataset.loader import IOVNBDDataset, SynchronizedTrip
from src.calibration.aligner import InVehicleAligner
from src.filters.prefilter import VibrationPreFilter
from src.dataset.window_dataset import IMUSpeedWindowDataset
from src.fusion.engine import IntelligentDeadReckoningEngine
from src.map_matching.road_network import RoadNetwork, RoadSegment


def latlon_to_meters(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    """Local flat-earth projection to convert lat/lon to local NED meters (East, North)."""
    R = 6378137.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    north = R * dlat
    east = R * dlon * np.cos(np.radians(lat0))
    return east, north


def meters_to_latlon(east: float, north: float, lat0: float, lon0: float):
    """Convert local NED meters back to lat/lon."""
    R = 6378137.0
    dlat = north / R
    dlon = east / (R * np.cos(np.radians(lat0)))
    lat = lat0 + np.degrees(dlat)
    lon = lon0 + np.degrees(dlon)
    return float(lat), float(lon)


class NavigationSession:
    def __init__(self, trip_info: Dict[str, str], data_loader: IOVNBDDataset):
        self.trip_info = trip_info
        self.data_loader = data_loader
        
        # 1. Load synchronized trip
        self.trip: SynchronizedTrip = data_loader.load_trip(
            trip_info["phone_csv"], trip_info["vehicle_csv"], trip_id=f"{trip_info['driver']}_{trip_info['trip']}"
        )
        self.dt = self.trip.dt
        self.num_frames = len(self.trip.time_s)
        
        # 2. Local origin (Reference Point)
        self.lat0 = float(self.trip.gt_lat[0])
        self.lon0 = float(self.trip.gt_lon[0])
        
        # 3. Ground truth East/North positions
        gt_e, gt_n = latlon_to_meters(self.trip.gt_lat, self.trip.gt_lon, self.lat0, self.lon0)
        self.gt_pos_m = np.column_stack([gt_e, gt_n])
        
        # 4. In-Vehicle Sensor Alignment
        self.aligner = InVehicleAligner(min_accel_dynamics_thresh=0.4)
        init_len = min(600, self.num_frames)
        self.aligner.calibrate(
            acc=self.trip.acc[:init_len],
            gravity=self.trip.gravity[:init_len],
            speed_ref=self.trip.gt_speed_ms[:init_len]
        )
        self.acc_v = self.aligner.transform_vector(self.trip.acc)
        self.gyro_v = self.aligner.transform_vector(self.trip.gyro)
        
        # 5. Pre-filter & Road Texture Vibration
        self.prefilter = VibrationPreFilter(sampling_rate_hz=1.0 / self.dt)
        self.filtered = self.prefilter.process(self.acc_v, self.gyro_v, real_time=True)
        self.acc_clean = self.filtered.acc_clean
        self.gyro_clean = self.filtered.gyro_clean
        self.is_stationary = self.filtered.is_stationary
        
        hf_noise = np.linalg.norm(self.acc_v - self.acc_clean, axis=1)
        self.v_rms = pd.Series(hf_noise).rolling(20, min_periods=1).mean().values
        
        # 6. Feature Windows for SpeedNet
        self.X_windows, _ = IMUSpeedWindowDataset.extract_features_from_trip(
            self.trip, window_len=40, stride=1
        )
        self.pad_len = self.num_frames - len(self.X_windows)
        
        # 7. Build Road Network Graph for HMM Lane-Snapping
        self.road_net = RoadNetwork(cell_size=50.0)
        gt_coords = self.gt_pos_m
        chunk_step = 20
        for i in range(0, len(gt_coords) - chunk_step, chunk_step):
            poly = gt_coords[i : i + chunk_step + 1]
            seg = RoadSegment(
                segment_id=f"seg_{i}",
                start_node=f"node_{i}",
                end_node=f"node_{i+chunk_step}",
                polyline=poly,
                is_one_way=True,
                speed_limit_kmh=80.0
            )
            self.road_net.add_segment(seg)
            
        # 8. Initialize Fusion Engine
        self.engine = IntelligentDeadReckoningEngine(dt=self.dt)
        self.engine.set_road_network(self.road_net, trust_curvature=True)
        
        # State tracking
        self.current_frame = 0
        self.blackout_forced = False
        self.in_blackout = False
        self.blackout_start_pos = None
        self.total_blackout_distance = 0.0

    def toggle_blackout(self, forced: Optional[bool] = None) -> bool:
        """Toggle or explicitly set manual GPS blackout."""
        if forced is not None:
            self.blackout_forced = forced
        else:
            self.blackout_forced = not self.blackout_forced
            
        t_curr = float(self.trip.time_s[min(self.current_frame, self.num_frames - 1)])
        curr_gt = self.gt_pos_m[min(self.current_frame, self.num_frames - 1)]
        v_gt = float(self.trip.gt_speed_ms[min(self.current_frame, self.num_frames - 1)])
        h_gt = np.radians(float(self.trip.gt_heading_deg[min(self.current_frame, self.num_frames - 1)]))

        if self.blackout_forced:
            if not self.in_blackout:
                self.in_blackout = True
                self.blackout_start_pos = curr_gt.copy()
                self.total_blackout_distance = 0.0
                
                # Pre-outage dynamic road calibration
                lookback = max(0, self.current_frame - 200)
                if self.current_frame - lookback > 10:
                    self.engine.calibrate_pre_outage(
                        self.v_rms[lookback:self.current_frame],
                        self.trip.gt_speed_ms[lookback:self.current_frame]
                    )
                    
                self.engine.on_gnss_lost(
                    time_s=t_curr,
                    p_entry=curr_gt,
                    v_entry=v_gt,
                    heading_entry_rad=h_gt
                )
        else:
            if self.in_blackout:
                self.in_blackout = False
                self.engine.on_gnss_restored(
                    time_s=t_curr,
                    p_gnss=curr_gt,
                    v_gnss=v_gt,
                    heading_gnss_rad=h_gt
                )
        return self.blackout_forced

    def step(self, step_size: int = 10) -> Dict[str, Any]:
        """Advance simulation by step_size frames (at 10Hz/100Hz)."""
        if self.current_frame >= self.num_frames:
            return {"finished": True, "frame": self.current_frame, "total_frames": self.num_frames}

        end_frame = min(self.current_frame + step_size, self.num_frames)
        
        for idx in range(self.current_frame, end_frame):
            t = float(self.trip.time_s[idx])
            w_yaw = float(self.gyro_clean[idx, 0])
            v_rms_i = float(self.v_rms[idx])
            stat = bool(self.is_stationary[idx])
            
            # SpeedNet window
            feat_idx = idx - self.pad_len
            feat = self.X_windows[feat_idx] if (0 <= feat_idx < len(self.X_windows)) else None
            
            if self.blackout_forced:
                pos, spd, head = self.engine.step_dead_reckoning(
                    yaw_rate_rad_s=w_yaw,
                    features_window=feat,
                    v_rms_instant=v_rms_i,
                    is_stationary=stat,
                    time_s=t
                )
                self.total_blackout_distance += spd * self.dt
            else:
                # GNSS Active Tracking
                gt_pt = self.gt_pos_m[idx]
                gt_spd = float(self.trip.gt_speed_ms[idx])
                gt_hd = np.radians(float(self.trip.gt_heading_deg[idx]))
                self.engine.current_pos = gt_pt.copy()
                self.engine.current_speed_ms = gt_spd
                self.engine.current_heading_rad = gt_hd

        self.current_frame = end_frame
        curr_idx = min(self.current_frame - 1, self.num_frames - 1)
        
        # Coordinates
        gt_east, gt_north = self.gt_pos_m[curr_idx]
        gt_lat, gt_lon = meters_to_latlon(gt_east, gt_north, self.lat0, self.lon0)
        
        ins_east, ins_north = self.engine.current_pos
        ins_lat, ins_lon = meters_to_latlon(ins_east, ins_north, self.lat0, self.lon0)
        
        snapped_pos = self.engine.get_lane_matched_position()
        snapped_lat, snapped_lon = meters_to_latlon(snapped_pos[0], snapped_pos[1], self.lat0, self.lon0)
        
        # Drift metrics
        horizontal_drift = float(np.linalg.norm([ins_east - gt_east, ins_north - gt_north]))
        drift_percent = 0.0
        if self.in_blackout and self.total_blackout_distance > 5.0:
            drift_percent = (horizontal_drift / self.total_blackout_distance) * 100.0
            
        speed_kmh = float(self.engine.current_speed_ms * 3.6)
        heading_deg = float(np.degrees(self.engine.current_heading_rad) % 360.0)
        
        # Uncertainty estimate (increases during blackout, collapses on GNSS)
        uncertainty = 0.25 if not self.blackout_forced else min(15.0, 0.25 + 0.015 * self.total_blackout_distance)

        return {
            "finished": self.current_frame >= self.num_frames,
            "frame": self.current_frame,
            "total_frames": self.num_frames,
            "progress_percent": round((self.current_frame / self.num_frames) * 100.0, 1),
            "blackout_active": self.blackout_forced,
            "state_mode": "DEAD_RECKONING" if self.blackout_forced else "GNSS_AIDED",
            "speed_kmh": round(speed_kmh, 1),
            "heading_deg": round(heading_deg, 1),
            "horizontal_drift_m": round(horizontal_drift, 2),
            "drift_percent": round(drift_percent, 2),
            "blackout_distance_m": round(self.total_blackout_distance, 1),
            "sih_compliant": bool(drift_percent < 10.0 or not self.in_blackout),
            "current_position": {
                "ins": {"lat": ins_lat, "lon": ins_lon},
                "gt": {"lat": gt_lat, "lon": gt_lon},
                "snapped": {"lat": snapped_lat, "lon": snapped_lon}
            },
            "imu": {
                "accel": [float(a) for a in self.acc_clean[curr_idx]],
                "gyro": [float(g) for g in self.gyro_clean[curr_idx]]
            },
            "uncertainty_sigma_m": round(uncertainty, 2)
        }

    def reset(self):
        """Reset session to beginning."""
        self.current_frame = 0
        self.blackout_forced = False
        self.in_blackout = False
        self.blackout_start_pos = None
        self.total_blackout_distance = 0.0
        self.engine = IntelligentDeadReckoningEngine(dt=self.dt)
        self.engine.set_road_network(self.road_net, trust_curvature=True)


class SessionManager:
    """Manages active navigation simulation sessions."""
    def __init__(self, root_dir: Optional[str] = None):
        if root_dir is None:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IO-VNBD"))
        self.root_dir = root_dir
        self.data_loader = IOVNBDDataset(root_dir=root_dir)
        self.available_trips = self.data_loader.list_available_trips()
        self.active_session: Optional[NavigationSession] = None

    def list_available_trips(self) -> List[str]:
        """List human-readable trip identifiers."""
        return [f"{t['driver']} - {t['trip']}" for t in self.available_trips]

    def start_session(self, trip_id: str) -> Dict[str, Any]:
        """Start a new session for the chosen trip."""
        match = None
        for t in self.available_trips:
            name = f"{t['driver']} - {t['trip']}"
            if name == trip_id or trip_id in name:
                match = t
                break
        if match is None and self.available_trips:
            match = self.available_trips[0]
            
        self.active_session = NavigationSession(trip_info=match, data_loader=self.data_loader)
        return {
            "trip_id": f"{match['driver']} - {match['trip']}",
            "num_frames": self.active_session.num_frames,
            "origin": {"lat": self.active_session.lat0, "lon": self.active_session.lon0}
        }

    def get_session(self) -> NavigationSession:
        if self.active_session is None:
            if self.available_trips:
                self.start_session(f"{self.available_trips[0]['driver']} - {self.available_trips[0]['trip']}")
            else:
                raise ValueError("No available trips found in IO-VNBD dataset.")
        return self.active_session
