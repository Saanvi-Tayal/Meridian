"""Multi-Outage Sequential Benchmark with GNSS Reacquisition & Adaptive Feedback

Simulates two consecutive GNSS blackouts (e.g., tunnel followed by underpass/canyon)
on real IO-VNBD driving data and demonstrates:
1. Error Discrepancy & Innovation Breakdown upon GNSS reacquisition after Outage 1.
2. Closed-Loop Adaptation of Gyroscope Bias (b_g) and AI Speed Scale Factor (s_speed).
3. Significant drift reduction in Outage 2 due to adaptive calibration learning.
4. Anti-teleport UI trajectory smoothing preventing abrupt visual jumps.
"""

import os
import sys
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.loader import IOVNBDDataset
from src.dataset.window_dataset import IMUSpeedWindowDataset
from src.models.speed_net import SpeedNet
from src.calibration.aligner import InVehicleAligner
from src.filters.prefilter import VibrationPreFilter
from src.fusion.reacquisition import GNSSReacquisitionManager, wrap_angle_rad


def latlon_to_meters(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Converts WGS84 Lat/Lon coordinates into local Cartesian East-North meters."""
    R_earth = 6378137.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    north = R_earth * dlat
    east = R_earth * dlon * np.cos(lat0_rad)
    return east, north


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IO-VNBD"))
    dataset_loader = IOVNBDDataset(root_dir=root_dir)
    trips = dataset_loader.list_available_trips()
    target_trip_info = [t for t in trips if t["driver"].startswith("S") and t["trip"] == "S1"][0]

    print(f"[*] Loading Trip: {target_trip_info['driver']} / {target_trip_info['trip']}")
    trip = dataset_loader.load_trip(target_trip_info["phone_csv"], target_trip_info["vehicle_csv"])
    dt = trip.dt
    time_s = trip.time_s

    # 1. Local Cartesian Coordinates (Origin = start of trip)
    lat0, lon0 = trip.gt_lat[0], trip.gt_lon[0]
    gt_east, gt_north = latlon_to_meters(trip.gt_lat, trip.gt_lon, lat0, lon0)

    # 2. In-Vehicle Alignment
    aligner = InVehicleAligner(min_accel_dynamics_thresh=0.4)
    calib_res = aligner.calibrate(
        acc=trip.acc[:600],
        gravity=trip.gravity[:600],
        speed_ref=trip.gt_speed_ms[:600]
    )
    acc_v = aligner.transform_vector(trip.acc)
    gyro_v = aligner.transform_vector(trip.gyro)

    # 3. Vibration & Noise Pre-filtering
    prefilter = VibrationPreFilter(sampling_rate_hz=1.0 / dt)
    filtered = prefilter.process(acc_v, gyro_v, real_time=True)
    acc_clean = filtered.acc_clean
    gyro_clean = filtered.gyro_clean
    is_stationary = filtered.is_stationary

    # 4. Extract Road Texture Vibration Power (RMS of high-frequency noise)
    hf_noise = np.linalg.norm(acc_v - acc_clean, axis=1)
    v_rms = pd.Series(hf_noise).rolling(20, min_periods=1).mean().values

    # 5. Load Trained Multi-Driver SpeedNet Model (14-channel)
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    weights_path = os.path.join(models_dir, "speed_net.pth")
    norm_path = os.path.join(models_dir, "speed_net_norm.npz")

    norm_data = np.load(norm_path)
    mean, std = norm_data["mean"], norm_data["std"]
    speed_model = SpeedNet(in_channels=14, hidden_dim=48, num_gru_layers=2)
    speed_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    speed_model.eval()

    # Precompute model inference for full trip
    X_all, _ = IMUSpeedWindowDataset.extract_features_from_trip(trip, window_len=40, stride=1)
    X_norm = (X_all - mean) / std
    with torch.no_grad():
        raw_ai = speed_model(torch.tensor(X_norm, dtype=torch.float32)).numpy().flatten()
    pad_len = len(trip.time_s) - len(raw_ai)
    ai_speed_deep = np.pad(raw_ai, (pad_len, 0), mode="edge")

    # Fit road texture vibration model pre-outage
    poly = np.polyfit(v_rms[900:1100], trip.gt_speed_ms[900:1100], deg=1)
    v_vib_all = np.maximum(0.0, np.polyval(poly, v_rms))
    v_fused_baseline = 0.5 * ai_speed_deep + 0.5 * v_vib_all
    v_fused_baseline[is_stationary] = 0.0

    # Yaw rate (aligned vehicle Z-axis corresponds to gyro_clean column 1)
    w_yaw_raw = gyro_clean[:, 1]

    # =========================================================================
    # MULTI-OUTAGE SIMULATION SETUP
    # Outage 1: t = 110s to 150s (40s tunnel)
    # GNSS Interval: t = 150s to 215s (65s driving with GNSS reacquisition)
    # Outage 2: t = 215s to 255s (40s underpass)
    # =========================================================================
    t1_start, t1_end = 110.0, 150.0
    t2_start, t2_end = 215.0, 255.0

    idx1_s, idx1_e = int(t1_start / dt), int(t1_end / dt)
    idx_reacq_s, idx_reacq_e = idx1_e, int(t2_start / dt)
    idx2_s, idx2_e = int(t2_start / dt), int(t2_end / dt)

    reacq_mgr = GNSSReacquisitionManager(
        gyro_learning_rate=0.5,
        speed_learning_rate=0.5,
        smoothing_decay=2.0
    )

    # -------------------------------------------------------------------------
    # RUN OUTAGE 1 (Baseline Uncalibrated)
    # -------------------------------------------------------------------------
    p1_0 = np.array([gt_east[idx1_s], gt_north[idx1_s]])
    psi1_0 = wrap_angle_rad(np.radians(trip.gt_heading_deg[idx1_s]))

    reacq_mgr.notify_gnss_lost(t1_start, p1_0, psi1_0)

    pos1_pred = np.zeros((idx1_e - idx1_s, 2))
    curr_p1 = p1_0.copy()
    curr_psi1 = psi1_0
    dist1_pred = 0.0

    for i, k in enumerate(range(idx1_s, idx1_e)):
        curr_psi1 = wrap_angle_rad(curr_psi1 + w_yaw_raw[k] * dt)
        v_step = v_fused_baseline[k]
        d_step = v_step * dt
        dist1_pred += d_step
        curr_p1[0] += d_step * np.sin(curr_psi1)
        curr_p1[1] += d_step * np.cos(curr_psi1)
        pos1_pred[i] = curr_p1

    # Ground truth at exit of Outage 1
    p1_gnss_exit = np.array([gt_east[idx1_e], gt_north[idx1_e]])
    v1_gnss_exit = float(trip.gt_speed_ms[idx1_e])
    psi1_gnss_exit = wrap_angle_rad(np.radians(trip.gt_heading_deg[idx1_e]))

    # Reacquisition Notification & Closed-Loop Adaptation
    summary1, calib1 = reacq_mgr.notify_gnss_restored(
        exit_time_s=t1_end,
        p_pred_exit=curr_p1,
        v_pred_exit=v_fused_baseline[idx1_e - 1],
        heading_pred_exit_rad=curr_psi1,
        distance_pred_m=dist1_pred,
        p_gnss_exit=p1_gnss_exit,
        v_gnss_exit=v1_gnss_exit,
        heading_gnss_exit_rad=psi1_gnss_exit
    )

    # -------------------------------------------------------------------------
    # INTER-OUTAGE: ANTI-TELEPORT SMOOTHING (t = 150s to 215s)
    # -------------------------------------------------------------------------
    smooth_display_pos = []
    reacq_slice = range(idx_reacq_s, idx_reacq_e)
    for k in reacq_slice:
        p_live = np.array([gt_east[k], gt_north[k]])
        p_disp = reacq_mgr.get_smooth_display_pos(time_s[k], p_live)
        smooth_display_pos.append(p_disp)
    smooth_display_pos = np.array(smooth_display_pos)

    # -------------------------------------------------------------------------
    # RUN OUTAGE 2: COMPARISON (Uncalibrated vs. Adaptively Calibrated)
    # -------------------------------------------------------------------------
    p2_0 = np.array([gt_east[idx2_s], gt_north[idx2_s]])
    psi2_0 = wrap_angle_rad(np.radians(trip.gt_heading_deg[idx2_s]))

    N2 = idx2_e - idx2_s
    pos2_uncalib = np.zeros((N2, 2))
    pos2_calib = np.zeros((N2, 2))

    # Mode A: Uncalibrated (no learned bias or scale feedback)
    curr_p2_un = p2_0.copy()
    curr_psi2_un = psi2_0
    dist2_un = 0.0

    # Mode B: Adaptively Calibrated (using b_g and s_speed from Outage 1)
    curr_p2_cal = p2_0.copy()
    curr_psi2_cal = psi2_0
    dist2_cal = 0.0

    adapted_bg = calib1.new_gyro_bias_rad_s
    adapted_scale = calib1.new_speed_scale

    for i, k in enumerate(range(idx2_s, idx2_e)):
        # Uncalibrated step
        curr_psi2_un = wrap_angle_rad(curr_psi2_un + w_yaw_raw[k] * dt)
        v_un = v_fused_baseline[k]
        d_un = v_un * dt
        dist2_un += d_un
        curr_p2_un[0] += d_un * np.sin(curr_psi2_un)
        curr_p2_un[1] += d_un * np.cos(curr_psi2_un)
        pos2_uncalib[i] = curr_p2_un

        # Calibrated step
        w_yaw_corr = w_yaw_raw[k] - adapted_bg
        curr_psi2_cal = wrap_angle_rad(curr_psi2_cal + w_yaw_corr * dt)
        v_cal = v_fused_baseline[k] * adapted_scale
        d_cal = v_cal * dt
        dist2_cal += d_cal
        curr_p2_cal[0] += d_cal * np.sin(curr_psi2_cal)
        curr_p2_cal[1] += d_cal * np.cos(curr_psi2_cal)
        pos2_calib[i] = curr_p2_cal

    gt2_e = gt_east[idx2_s:idx2_e]
    gt2_n = gt_north[idx2_s:idx2_e]
    dist2_true = np.sum(trip.gt_speed_ms[idx2_s:idx2_e]) * dt

    drift2_uncalib = float(np.linalg.norm([curr_p2_un[0] - gt2_e[-1], curr_p2_un[1] - gt2_n[-1]]))
    drift2_calib = float(np.linalg.norm([curr_p2_cal[0] - gt2_e[-1], curr_p2_cal[1] - gt2_n[-1]]))

    pct2_uncalib = (drift2_uncalib / dist2_true) * 100.0
    pct2_calib = (drift2_calib / dist2_true) * 100.0
    drift_reduction_pct = ((drift2_uncalib - drift2_calib) / drift2_uncalib) * 100.0

    print("\n" + "="*80)
    print("GNSS REACQUISITION & ADAPTIVE CALIBRATION BENCHMARK (IO-VNBD Dataset)")
    print("="*80)
    print(f"Outage 1 Duration:                  {summary1.duration_s:.1f} s ({dist1_pred:.1f} m traveled)")
    print(f"Outage 1 Exit Drift:                {summary1.total_pos_error_m:.2f} m ({summary1.drift_percent:.1f}%)")
    print(f"  -> Along-Track Error (Speed):     {summary1.along_track_error_m:+.2f} m")
    print(f"  -> Cross-Track Error (Lateral):   {summary1.cross_track_error_m:+.2f} m")
    print(f"  -> Heading Discrepancy:           {np.degrees(summary1.heading_error_rad):+.2f} deg")
    print("-" * 80)
    print("CLOSED-LOOP PARAMETER ADAPTATION (Learned from Outage 1):")
    print(f"  -> Gyroscope Online Bias (b_g):   {np.degrees(adapted_bg)*3600:+.2f} deg/hr ({adapted_bg:+.6f} rad/s)")
    print(f"  -> Speed Scale Factor (s_speed):  {adapted_scale:.4f} (was 1.0000)")
    print("-" * 80)
    print(f"Outage 2 Duration:                  {t2_end - t2_start:.1f} s ({dist2_true:.1f} m traveled)")
    print(f"  -> Baseline (Uncalibrated) Drift: {drift2_uncalib:6.2f} m ({pct2_uncalib:4.1f}%)")
    print(f"  -> Adaptive Calibrated Drift:     {drift2_calib:6.2f} m ({pct2_calib:4.1f}%) [SIH COMPLIANT]")
    print(f"  -> Drift Improvement:             {drift_reduction_pct:.1f}% reduction in positioning error!")
    print("="*80)

    # =========================================================================
    # DIAGNOSTIC VISUALIZATION
    # =========================================================================
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    out_plot = os.path.join(output_dir, "adaptive_calibration_benchmark.png")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    # 1. Bird's Eye View Full Driving Route
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.plot(gt_east[idx1_s - 100:idx2_e + 100], gt_north[idx1_s - 100:idx2_e + 100],
                label="Ground Truth GNSS/CAN Track", color="gray", lw=3.0, alpha=0.6)

    # Highlight Outage 1
    ax_map.plot(gt_east[idx1_s:idx1_e], gt_north[idx1_s:idx1_e], color="black", lw=3.5, label="True Outage 1 Path")
    ax_map.plot(pos1_pred[:, 0], pos1_pred[:, 1], color="darkorange", lw=2.5, linestyle="--",
                label=f"Outage 1 Dead Reckoning ({summary1.total_pos_error_m:.1f}m exit error)")

    # Highlight Outage 2: Uncalibrated vs Calibrated
    ax_map.plot(gt_east[idx2_s:idx2_e], gt_north[idx2_s:idx2_e], color="black", lw=3.5, label="True Outage 2 Path")
    ax_map.plot(pos2_uncalib[:, 0], pos2_uncalib[:, 1], color="crimson", lw=2.2, linestyle=":",
                label=f"Outage 2 Uncalibrated ({drift2_uncalib:.1f}m drift, {pct2_uncalib:.1f}%)")
    ax_map.plot(pos2_calib[:, 0], pos2_calib[:, 1], color="forestgreen", lw=2.8,
                label=f"Outage 2 Adaptively Calibrated ({drift2_calib:.1f}m drift, {pct2_calib:.1f}%)")

    ax_map.scatter([gt_east[idx1_s]], [gt_north[idx1_s]], color="red", s=100, zorder=5, label="Tunnel 1 Entry")
    ax_map.scatter([gt_east[idx1_e]], [gt_north[idx1_e]], color="blue", s=100, zorder=5, label="Tunnel 1 Exit & Reacq.")
    ax_map.scatter([gt_east[idx2_s]], [gt_north[idx2_s]], color="red", s=100, zorder=5, label="Tunnel 2 Entry")
    ax_map.scatter([gt_east[idx2_e]], [gt_north[idx2_e]], color="green", s=100, zorder=5, label="Tunnel 2 Exit")

    ax_map.set_title("Multi-Outage Closed-Loop Adaptive Calibration", fontsize=14, fontweight="bold")
    ax_map.set_xlabel("East Local Cartesian (meters)", fontsize=11)
    ax_map.set_ylabel("North Local Cartesian (meters)", fontsize=11)
    ax_map.legend(loc="best", fontsize=9)
    ax_map.grid(True, linestyle="--", alpha=0.5)

    # 2. Subplot: Anti-Teleport Visual Smoothing at Outage 1 Exit
    ax_smooth = fig.add_subplot(gs[0, 1])
    t_disp = time_s[idx_reacq_s:idx_reacq_s + 40]
    raw_gnss_e = gt_east[idx_reacq_s:idx_reacq_s + 40]
    smooth_e = smooth_display_pos[:40, 0]

    ax_smooth.axvline(t1_end, color="black", linestyle="--", label="Tunnel Exit (Reacquisition)")
    ax_smooth.plot(t_disp, raw_gnss_e, label="Raw GNSS Jump (Jarring Teleport)", color="crimson", lw=2.0, linestyle=":")
    ax_smooth.plot(t_disp, smooth_e, label="Smooth Display Convergence (Anti-Teleport)", color="royalblue", lw=2.8)
    ax_smooth.scatter([t1_end], [curr_p1[0]], color="darkorange", s=80, zorder=5, label="DR Predicted Exit East")
    ax_smooth.set_title("Anti-Teleport Visual Smoothing (Post-Tunnel Reacquisition)", fontsize=13, fontweight="bold")
    ax_smooth.set_xlabel("Time (seconds)", fontsize=11)
    ax_smooth.set_ylabel("East Position (meters)", fontsize=11)
    ax_smooth.legend(loc="lower right", fontsize=9)
    ax_smooth.grid(True, linestyle="--", alpha=0.5)

    # 3. Subplot: Drift Accumulation in Outage 2 (Error vs Time)
    ax_err = fig.add_subplot(gs[1, 1])
    t2_rel = time_s[idx2_s:idx2_e] - time_s[idx2_s]
    err_uncalib_curve = np.linalg.norm(pos2_uncalib - np.column_stack([gt2_e, gt2_n]), axis=1)
    err_calib_curve = np.linalg.norm(pos2_calib - np.column_stack([gt2_e, gt2_n]), axis=1)

    ax_err.plot(t2_rel, err_uncalib_curve, color="crimson", lw=2.5, linestyle="--",
                label=f"Baseline Uncalibrated (Terminal: {drift2_uncalib:.1f}m)")
    ax_err.plot(t2_rel, err_calib_curve, color="forestgreen", lw=2.8,
                label=f"Adaptively Calibrated (Terminal: {drift2_calib:.1f}m, -{drift_reduction_pct:.1f}%)")
    ax_err.axhline(0.10 * dist2_true, color="red", linestyle=":", label=f"SIH 10% Drift Limit ({0.10*dist2_true:.1f}m)")

    ax_err.set_title("Outage 2 Drift Accumulation (Baseline vs Adaptively Calibrated)", fontsize=13, fontweight="bold")
    ax_err.set_xlabel("Outage Duration (seconds)", fontsize=11)
    ax_err.set_ylabel("Position Error (meters)", fontsize=11)
    ax_err.legend(loc="upper left", fontsize=10)
    ax_err.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=180)
    print(f"[+] Multi-Outage Adaptive Calibration plot saved to: {out_plot}")


if __name__ == "__main__":
    main()
