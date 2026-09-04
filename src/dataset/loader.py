"""IO-VNBD Dataset Loader & Preprocessor

Parses, cleans, and time-synchronizes smartphone IMU telemetry and vehicle CAN-bus
ground truth from the Inertial and Odometry Benchmark Dataset (IO-VNBD).
"""

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd


@dataclass
class SynchronizedTrip:
    """Synchronized container for a single driving session."""
    trip_id: str
    time_s: np.ndarray             # Time vector in seconds (relative to trip start)
    dt: float                      # Average sampling period (e.g. 0.1s for 10Hz)
    
    # Phone Sensor Telemetry (Sensor Frame)
    acc: np.ndarray                # Shape (N, 3): [acc_x, acc_y, acc_z] in m/s^2
    gyro: np.ndarray               # Shape (N, 3): [yaw_rate, pitch_rate, roll_rate] in rad/s
    gravity: np.ndarray            # Shape (N, 3): [grav_x, grav_y, grav_z] in m/s^2
    mag: Optional[np.ndarray]      # Shape (N, 3): [mag_x, mag_y, mag_z] in uT
    
    # Phone GNSS (when available)
    phone_lat: np.ndarray          # Latitude in degrees
    phone_lon: np.ndarray          # Longitude in degrees
    phone_speed_ms: np.ndarray     # Speed in m/s
    phone_accuracy: np.ndarray     # Accuracy in meters
    phone_satellites: np.ndarray   # Available satellites
    
    # Vehicle CAN-bus Ground Truth (Synchronized to phone timestamps)
    gt_speed_ms: np.ndarray        # Ground truth forward velocity in m/s
    gt_lat: np.ndarray             # Ground truth latitude
    gt_lon: np.ndarray             # Ground truth longitude
    gt_heading_deg: np.ndarray     # Ground truth heading in degrees (0-360)
    gt_long_accel_ms2: np.ndarray  # Ground truth forward acceleration in m/s^2
    gt_yaw_rate_rads: np.ndarray   # Ground truth vehicle yaw rate in rad/s


class IOVNBDDataset:
    """Manages loading and querying IO-VNBD dataset records."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.categorised_dir = os.path.join(
            root_dir, "Synchronised V abd S datasets", "Categorised IOVNB Dataset"
        )
        if not os.path.exists(self.categorised_dir):
            raise FileNotFoundError(f"Categorised dataset path not found at: {self.categorised_dir}")

    def list_available_trips(self) -> List[Dict[str, str]]:
        """Finds all paired phone (S-*.csv) and vehicle (V-*.csv) files."""
        trips = []
        for driver in os.listdir(self.categorised_dir):
            driver_dir = os.path.join(self.categorised_dir, driver)
            if not os.path.isdir(driver_dir):
                continue
            for trip in os.listdir(driver_dir):
                trip_dir = os.path.join(driver_dir, trip)
                if not os.path.isdir(trip_dir):
                    continue
                s_files = [f for f in os.listdir(trip_dir) if f.startswith("S-") and f.endswith(".csv")]
                v_files = [f for f in os.listdir(trip_dir) if f.startswith("V-") and f.endswith(".csv")]
                if s_files and v_files:
                    trips.append({
                        "driver": driver,
                        "trip": trip,
                        "phone_csv": os.path.join(trip_dir, s_files[0]),
                        "vehicle_csv": os.path.join(trip_dir, v_files[0]),
                    })
        return trips

    @staticmethod
    def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Removes encoding anomalies from IO-VNBD CSV headers."""
        cleaned = {}
        for col in df.columns:
            # Strip non-ascii chars and extra spaces
            norm = re.sub(r'[^\x00-\x7F]+', '_', col).strip()
            norm = re.sub(r'\s+', '_', norm)
            cleaned[col] = norm
        return df.rename(columns=cleaned)

    def load_trip(self, phone_csv: str, vehicle_csv: str, trip_id: str = "trip") -> SynchronizedTrip:
        """Loads and synchronizes a phone-vehicle paired dataset."""
        s_df = pd.read_csv(phone_csv, encoding="latin1")
        v_df = pd.read_csv(vehicle_csv, encoding="latin1")

        s_df = self._clean_column_names(s_df)
        v_df = self._clean_column_names(v_df)

        # 1. Parse Phone Timestamps
        # 'TIME_SINCE_START__ms_' is available in milliseconds
        time_col = [c for c in s_df.columns if "TIME_SINCE_START" in c][0]
        phone_time_s = s_df[time_col].values.astype(float) / 1000.0
        phone_time_s = phone_time_s - phone_time_s[0]

        # 2. Extract Phone IMU Features
        acc_x = s_df[[c for c in s_df.columns if "ACCELEROMETER_X" in c][0]].values.astype(float)
        acc_y = s_df[[c for c in s_df.columns if "ACCELEROMETER_Y" in c][0]].values.astype(float)
        acc_z = s_df[[c for c in s_df.columns if "ACCELEROMETER_Z" in c][0]].values.astype(float)
        acc = np.column_stack([acc_x, acc_y, acc_z])

        # Phone Gyroscope (Yaw, Pitch, Roll in rad/s)
        gyro_yaw = s_df[[c for c in s_df.columns if "GYROSCOPE_Yaw" in c or ("GYROSCOPE" in c and "Yaw" in c)][0]].values.astype(float)
        gyro_pitch = s_df[[c for c in s_df.columns if "GYROSCOPE_Pitch" in c or ("GYROSCOPE" in c and "Pitch" in c)][0]].values.astype(float)
        gyro_roll = s_df[[c for c in s_df.columns if "GYROSCOPE_Roll" in c or ("GYROSCOPE" in c and "Roll" in c)][0]].values.astype(float)
        gyro = np.column_stack([gyro_yaw, gyro_pitch, gyro_roll])

        # Phone Gravity vector
        grav_x = s_df[[c for c in s_df.columns if "GRAVITY_X" in c][0]].values.astype(float)
        grav_y = s_df[[c for c in s_df.columns if "GRAVITY_Y" in c][0]].values.astype(float)
        grav_z = s_df[[c for c in s_df.columns if "GRAVITY_Z" in c][0]].values.astype(float)
        gravity = np.column_stack([grav_x, grav_y, grav_z])

        # Phone Magnetometer (if present)
        mag_cols = [c for c in s_df.columns if "MAGNETIC_FIELD" in c]
        if len(mag_cols) >= 3:
            mag = np.column_stack([
                s_df[mag_cols[0]].values.astype(float),
                s_df[mag_cols[1]].values.astype(float),
                s_df[mag_cols[2]].values.astype(float)
            ])
        else:
            mag = None

        # Phone GPS
        phone_lat = s_df[[c for c in s_df.columns if "GPS_LATITUDE" in c][0]].values.astype(float)
        phone_lon = s_df[[c for c in s_df.columns if "GPS_LONGITUDE" in c][0]].values.astype(float)
        phone_speed_ms = s_df[[c for c in s_df.columns if "GPS_SPEED" in c][0]].values.astype(float) / 3.6
        phone_acc = s_df[[c for c in s_df.columns if "GPS_ACCURACY" in c][0]].values.astype(float)
        sat_col = [c for c in s_df.columns if "GPS_SATELLITES" in c]
        if sat_col:
            # Format is often "27 / 28" - extract first integer
            sat_strs = s_df[sat_col[0]].astype(str)
            phone_sats = np.array([int(re.findall(r'\d+', s)[0]) if re.findall(r'\d+', s) else 0 for s in sat_strs])
        else:
            phone_sats = np.zeros(len(phone_lat))

        # 3. Parse Vehicle CAN Ground Truth Timestamps
        v_time_col = [c for c in v_df.columns if "Time_Since_Start" in c][0]
        v_time_s = v_df[v_time_col].values.astype(float)
        v_time_s = v_time_s - v_time_s[0]

        # Vehicle Velocity (convert km/h to m/s)
        v_speed_col = [c for c in v_df.columns if "Velocity" in c and "Vertical" not in c][0]
        v_speed_ms = v_df[v_speed_col].values.astype(float) / 3.6

        v_lat_col = [c for c in v_df.columns if "Latitude" in c][0]
        v_lon_col = [c for c in v_df.columns if "Longitude" in c][0]
        v_lat = v_df[v_lat_col].values.astype(float)
        v_lon = v_df[v_lon_col].values.astype(float)

        v_head_col = [c for c in v_df.columns if "Heading" in c][0]
        v_heading = v_df[v_head_col].values.astype(float)

        # Vehicle Longitudinal Acceleration (g to m/s^2)
        v_long_acc_col = [c for c in v_df.columns if "Longitudinal_Acceleration" in c][0]
        v_long_acc_ms2 = v_df[v_long_acc_col].values.astype(float) * 9.80665

        # Vehicle Yaw rate (deg/s to rad/s)
        v_yaw_col = [c for c in v_df.columns if "Yaw_Rate" in c][0]
        v_yaw_rate_rads = np.radians(v_df[v_yaw_col].values.astype(float))

        # 4. Synchronize Vehicle Ground Truth onto Phone Time Grid
        # Both logs start near time 0; use linear interpolation across phone_time_s
        # Bound interpolation to overlapping time range
        max_time = min(phone_time_s[-1], v_time_s[-1])
        valid_idx = phone_time_s <= max_time
        
        t_grid = phone_time_s[valid_idx]
        acc = acc[valid_idx]
        gyro = gyro[valid_idx]
        gravity = gravity[valid_idx]
        if mag is not None:
            mag = mag[valid_idx]
        phone_lat = phone_lat[valid_idx]
        phone_lon = phone_lon[valid_idx]
        phone_speed_ms = phone_speed_ms[valid_idx]
        phone_acc = phone_acc[valid_idx]
        phone_sats = phone_sats[valid_idx]

        gt_speed_ms = np.interp(t_grid, v_time_s, v_speed_ms)
        gt_lat = np.interp(t_grid, v_time_s, v_lat)
        gt_lon = np.interp(t_grid, v_time_s, v_lon)
        gt_heading_deg = np.interp(t_grid, v_time_s, v_heading)
        gt_long_accel_ms2 = np.interp(t_grid, v_time_s, v_long_acc_ms2)
        gt_yaw_rate_rads = np.interp(t_grid, v_time_s, v_yaw_rate_rads)

        dt = float(np.mean(np.diff(t_grid))) if len(t_grid) > 1 else 0.1

        return SynchronizedTrip(
            trip_id=trip_id,
            time_s=t_grid,
            dt=dt,
            acc=acc,
            gyro=gyro,
            gravity=gravity,
            mag=mag,
            phone_lat=phone_lat,
            phone_lon=phone_lon,
            phone_speed_ms=phone_speed_ms,
            phone_accuracy=phone_acc,
            phone_satellites=phone_sats,
            gt_speed_ms=gt_speed_ms,
            gt_lat=gt_lat,
            gt_lon=gt_lon,
            gt_heading_deg=gt_heading_deg,
            gt_long_accel_ms2=gt_long_accel_ms2,
            gt_yaw_rate_rads=gt_yaw_rate_rads,
        )
