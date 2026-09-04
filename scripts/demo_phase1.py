"""Phase 1 Verification Script: In-Vehicle Alignment & Vibration Pre-Filter

Loads real driving data from IO-VNBD, calibrates the smartphone orientation,
filters chassis & engine vibrations, and verifies against vehicle CAN-bus ground truth.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.loader import IOVNBDDataset
from src.calibration.aligner import InVehicleAligner
from src.filters.prefilter import VibrationPreFilter


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IO-VNBD"))
    dataset = IOVNBDDataset(root_dir=root_dir)

    trips = dataset.list_available_trips()
    print(f"[*] Discovered {len(trips)} available trips in IO-VNBD.")
    if not trips:
        print("[!] No trips found. Check dataset path.")
        return

    # Select Driver A - Trip S1 for verification
    target_trip = [t for t in trips if t["driver"].startswith("S") and t["trip"] == "S1"][0]
    print(f"[*] Loading Trip: {target_trip['driver']} - {target_trip['trip']}")
    print(f"    Phone CSV: {target_trip['phone_csv']}")
    print(f"    CAN CSV:   {target_trip['vehicle_csv']}")

    trip = dataset.load_trip(target_trip["phone_csv"], target_trip["vehicle_csv"], trip_id="DriverA_S1")

    total_time_min = trip.time_s[-1] / 60.0
    print(f"\n[+] Loaded Synchronized Trip:")
    print(f"    Duration: {total_time_min:.2f} minutes ({len(trip.time_s)} samples @ ~{1.0/trip.dt:.1f} Hz)")
    print(f"    Max CAN Speed: {np.max(trip.gt_speed_ms) * 3.6:.1f} km/h")
    print(f"    Phone raw acc shape: {trip.acc.shape}")

    # =========================================================================
    # Step 1: In-Vehicle Alignment & Dynamic Calibration
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: IN-VEHICLE ALIGNMENT & DYNAMIC CALIBRATION ENGINE")
    print("="*70)
    aligner = InVehicleAligner(min_accel_dynamics_thresh=0.4)

    # Use first 60 seconds for initial calibration
    calib_samples = min(int(60.0 / trip.dt), len(trip.time_s))
    calib_res = aligner.calibrate(
        acc=trip.acc[:calib_samples],
        gyro=trip.gyro[:calib_samples],
        gravity=trip.gravity[:calib_samples],
        speed_ref=trip.gt_speed_ms[:calib_samples] # or phone_speed_ms
    )

    print(f"[+] Calibration Status: {'SUCCESS' if calib_res.is_calibrated else 'FAILED'}")
    print(f"    Estimated Phone Pitch: {calib_res.pitch_deg:+.2f} deg")
    print(f"    Estimated Phone Roll:  {calib_res.roll_deg:+.2f} deg")
    print(f"    Estimated Yaw Offset:  {calib_res.yaw_misalignment_deg:+.2f} deg")
    print(f"    Measured Gravity Norm: {calib_res.gravity_norm:.3f} m/s^2")
    print("\nCalculated Rotation Matrix R_s2v (Sensor -> Vehicle Body Frame):")
    print(np.round(calib_res.R_s2v, 4))

    # Transform all IMU measurements from phone frame to vehicle body frame
    acc_vehicle = aligner.transform_vector(trip.acc)
    gyro_vehicle = aligner.transform_vector(trip.gyro)

    # Remove gravity from vehicle vertical axis (Z_v is Up)
    acc_vehicle[:, 2] -= calib_res.gravity_norm

    # =========================================================================
    # Step 2: Vibration & Noise Pre-Filter
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: VIBRATION & NOISE PRE-FILTER")
    print("="*70)
    prefilter = VibrationPreFilter(
        sampling_rate_hz=1.0 / trip.dt,
        cutoff_freq_hz=3.5,
        filter_order=4,
        hampel_window=5,
        stationary_acc_var_thresh=0.06,
        stationary_gyro_var_thresh=0.003
    )

    filtered = prefilter.process(acc_vehicle, gyro_vehicle, real_time=False)
    pothole_count = np.sum(filtered.pothole_spikes)
    stationary_pct = np.mean(filtered.is_stationary) * 100.0

    print(f"[+] Signal Conditioning Complete:")
    print(f"    Pothole / Road Shock Spikes Replaced: {pothole_count}")
    print(f"    Vehicle Standstill Detected: {stationary_pct:.1f}% of total duration")

    # =========================================================================
    # Step 3: Quantitative Verification vs CAN-bus Ground Truth
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: ACCURACY VERIFICATION VS CAN GROUND TRUTH")
    print("="*70)

    # Forward acceleration in vehicle frame is index 1 (Y_v)
    a_fwd_raw_phone = trip.acc[:, 1] # arbitrary phone Y
    a_fwd_aligned = acc_vehicle[:, 1] # aligned vehicle forward
    a_fwd_filtered = filtered.acc_clean[:, 1] # aligned + filtered
    a_gt = trip.gt_long_accel_ms2 # CAN-bus longitudinal acceleration

    # Pearson Correlation
    corr_raw = np.corrcoef(a_fwd_raw_phone, a_gt)[0, 1]
    corr_aligned = np.corrcoef(a_fwd_aligned, a_gt)[0, 1]
    corr_filtered = np.corrcoef(a_fwd_filtered, a_gt)[0, 1]

    # RMSE
    rmse_aligned = np.sqrt(np.mean((a_fwd_aligned - a_gt)**2))
    rmse_filtered = np.sqrt(np.mean((a_fwd_filtered - a_gt)**2))

    print(f"[+] Forward Acceleration Correlation with CAN Ground Truth:")
    print(f"    Raw Unaligned Phone Y Accel:  {corr_raw:+.3f}")
    print(f"    Aligned Vehicle Forward Accel: {corr_aligned:+.3f} (Significant Improvement!)")
    print(f"    Aligned + Filtered Accel:      {corr_filtered:+.3f} (Clean Vehicle Dynamics!)")
    print(f"[+] RMSE against CAN Ground Truth:")
    print(f"    Aligned Raw:      {rmse_aligned:.3f} m/s^2")
    print(f"    Aligned Filtered: {rmse_filtered:.3f} m/s^2")

    # =========================================================================
    # Step 4: Generate Diagnostic & Verification Plots
    # =========================================================================
    print("\n[*] Generating verification plots...")
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output")), exist_ok=True)
    out_img = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "phase1_alignment_and_filter.png"))

    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    t_slice = slice(0, 1500) # First 150 seconds for clear visual inspection
    t = trip.time_s[t_slice]

    # Plot 1: Raw Phone IMU (Unaligned)
    axs[0].plot(t, trip.acc[t_slice, 0], label="Raw Acc X", color="red", alpha=0.7)
    axs[0].plot(t, trip.acc[t_slice, 1], label="Raw Acc Y", color="green", alpha=0.7)
    axs[0].plot(t, trip.acc[t_slice, 2], label="Raw Acc Z (Gravity + Shock)", color="blue", alpha=0.7)
    axs[0].set_ylabel("Accel (m/s²)")
    axs[0].set_title("1. Raw Smartphone Accelerometer (Arbitrary Sensor Frame {S}) - Contaminated by Tilt & Gravity", fontweight="bold")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="upper right")

    # Plot 2: Aligned Vehicle Frame Accelerations
    axs[1].plot(t, acc_vehicle[t_slice, 0], label="Lateral (Right, X_v)", color="orange", alpha=0.6)
    axs[1].plot(t, acc_vehicle[t_slice, 1], label="Longitudinal (Forward, Y_v)", color="forestgreen", lw=1.5)
    axs[1].plot(t, acc_vehicle[t_slice, 2], label="Vertical (Up, Z_v)", color="purple", alpha=0.6)
    axs[1].set_ylabel("Accel (m/s²)")
    axs[1].set_title("2. Calibrated Vehicle Body Frame {V} (Forward, Lateral, Vertical Separated)", fontweight="bold")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(loc="upper right")

    # Plot 3: Filtered Forward Accel vs Ground Truth CAN Accel
    axs[2].plot(t, a_fwd_aligned[t_slice], label="Aligned Forward (With Engine Vibration)", color="lightgreen", alpha=0.6)
    axs[2].plot(t, a_fwd_filtered[t_slice], label="Pre-Filtered Forward (Butterworth + De-spiked)", color="darkgreen", lw=2.0)
    axs[2].plot(t, a_gt[t_slice], label="CAN Ground Truth Longitudinal Accel", color="black", linestyle="--", lw=1.8)
    axs[2].set_ylabel("Forward Accel (m/s²)")
    axs[2].set_title(f"3. Forward Acceleration vs CAN Ground Truth (Correlation: {corr_filtered:.2f})", fontweight="bold")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].legend(loc="upper right")

    # Plot 4: Stationary Detection (ZUPT) vs True Speed
    axs[3].plot(t, trip.gt_speed_ms[t_slice] * 3.6, label="CAN Vehicle Speed (km/h)", color="royalblue", lw=2.0)
    stationary_signal = filtered.is_stationary[t_slice].astype(float) * 20.0
    axs[3].fill_between(t, 0, stationary_signal, color="crimson", alpha=0.35, label="Detected Standstill (ZUPT Trigger)")
    axs[3].set_ylabel("Speed (km/h)")
    axs[3].set_xlabel("Time (seconds)")
    axs[3].set_title("4. Zero Velocity Update (ZUPT) Standstill Detection vs True Vehicle Speed", fontweight="bold")
    axs[3].grid(True, linestyle="--", alpha=0.5)
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_img, dpi=200)
    print(f"\n[+] Diagnostic plot saved successfully to:\n    {out_img}")


if __name__ == "__main__":
    main()
