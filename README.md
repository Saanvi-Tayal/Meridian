-

# 🚗 Smartphone Dead Reckoning & Navigation Engine

When a vehicle enters a tunnel, underground parking, or urban canyon, GPS completely disappears. Standard smartphone GPS freezes or drifts wildly. 

This project uses your phone's built-in sensors (accelerometer & gyroscope), neural networks, kinematic physics, and offline road matching to keep navigation accurate and smooth without any satellite signal.

---

## 📌 Module-by-Module Breakdown

### 1. In-Vehicle Alignment (Phone Placement Calibration)
* **What it does:** Drivers place phones in any orientation—vertical, horizontal, or tilted on a dashboard mount. This module figures out which way is "down" (gravity) and which way is "forward" (vehicle driving direction) within seconds, rotating sensor data into true vehicle coordinates.
* **Key File:** `src/calibration/aligner.py`
* **How it works:** Uses gravity estimation to find the vertical axis, and Principal Component Analysis (PCA) on acceleration spikes to lock onto the forward driving axis.

---

### 2. Noise & Vibration Pre-Filtering
* **What it does:** Smartphone sensors are cheap and pick up intense road vibrations, potholes, engine rumble, and sudden shocks. This module strips away the junk noise while keeping true vehicle movement.
* **Key File:** `src/filters/prefilter.py`
* **How it works:**
  * **Butterworth Low-Pass Filter:** Cuts off high-frequency vehicle body rattle above 5 Hz.
  * **Spike Rejector:** Catches sudden impacts (like speed bumps or potholes) so they don't corrupt navigation.
  * **ZUPT (Zero Velocity Update):** Detects when the car is stopped at traffic lights and forces speed to zero so position doesn't creep forward.

---

### 3. AI SpeedNet (Deep Learning Speed Estimator)
* **What it does:** Normally, dead reckoning doubles-integrates acceleration to find speed, which causes errors to blow up within seconds ($d = \frac{1}{2}at^2$). Instead, this deep learning model **directly predicts vehicle speed** from IMU vibration patterns and motion curves.
* **Key Files:** 
  * `src/models/speed_net.py` (Model architecture & ONNX export)
  * `scripts/train_speed_model.py` (Training pipeline)
  * `models/speed_net.onnx` (Lightweight exported model, ~394 KB)
* **How it works:** A 14-channel neural network combining **1D-CNN** (detects short-term engine/road vibration patterns) and **BiGRU** (tracks driving momentum over time). It runs in under 2 ms on standard mobile hardware.

---

### 4. 15-State Error-State Kalman Filter (ES-EKF)
* **What it does:** Fuses GPS, phone sensors, and AI speed predictions while tracking and correcting sensor drift in real time.
* **Key Files:** 
  * `src/fusion/eskf.py` (Kalman filter math)
  * `src/fusion/engine.py` (Main navigation fusion engine)
* **How it works:**
  * Tracks 15 physical states: 3D Position, 3D Velocity, 3D Heading/Orientation, 3D Accelerometer Bias, and 3D Gyroscope Bias.
  * Applies **Non-Holonomic Constraints (NHC)**: Enforces the physical rule that cars drive forward, not sideways or through the air ($v_y = 0, v_z = 0$).

---

### 5. Closed-Loop Reacquisition Calibration (Self-Learning)
* **What it does:** When you exit a tunnel and GPS reconnects, standard navigation apps "teleport" the car marker to the new GPS fix. This module compares where the AI *thought* you were against the real GPS exit position, learns from the mistake, and fixes the math for the next tunnel.
* **Key File:** `src/fusion/reacquisition.py`
* **How it works:**
  * **Error Breakdown:** Splits the exit error into heading error (gyro drift) vs. distance error (speed scale miscalculation).
  * **Closed-Loop Adaptation:** Automatically recalibrates gyro bias and speed scaling for future blackouts.
  * **Anti-Teleport Smoothing:** Slowly and smoothly transitions the car icon on screen over 1–2 seconds instead of jarring jumps.

---

### 6. Offline HMM Map Matching (Road Snapping)
* **What it does:** Vehicles are physically bound to paved roads. Even if dead reckoning drifts slightly into buildings or opposing traffic lanes, this module locks the car marker precisely to the actual road corridor.
* **Key Files:** 
  * `src/map_matching/road_network.py` (Fast spatial grid of road polylines, <0.2 ms lookup)
  * `src/map_matching/hmm_matcher.py` (Hidden Markov Model with Viterbi decoding)
  * `src/map_matching/curvature_feedback.py` (Road curve heading lock)
* **How it works:**
  * Evaluates both how close the car is to a road segment and whether the car is heading in the right direction (avoids snapping onto cross-bridges or reverse lanes).
  * Uses road curvature feedback to stop the gyro from accumulating turn errors inside long curves.

---

## 📊 Benchmark & Accuracy Results

Tested on real driving datasets from the **IO-VNBD Benchmark** across multiple drivers and phone mount styles:

| Method / System | Error Over ~850m Tunnel | Drift Ratio | Lateral Lane Error | Meets SIH Target? |
| :--- | :---: | :---: | :---: | :---: |
| **Traditional INS (Double Integration)** | $725.5\text{ m}$ | $> 85\%$ | Off into buildings | ❌ FAILED |
| **Pure Speed Integration** | $151.4\text{ m}$ | $17.9\%$ | Left the road | ❌ FAILED |
| **Our AI SpeedNet + ES-EKF** | **$23.4\text{ m}$** | **$2.7\%$** | Near road edge | ✅ **PASSED** |
| **Our Full Engine + HMM Road Snapping** | **$18.7\text{ m}$** | **$2.2\%$** | **$0.00\text{ m}$ (Exact Centerline)** | ✅ **PASSED (Best in Class)** |

* **SIH Competition Requirement:** $< 10\%$ drift over the outage.
* **Our Final Performance:** **$2.2\%$ drift** (over **$4.5\times$ better** than required).
* **Sequential Outage Learning:** The closed-loop reacquisition calibration reduced drift by another **$32.5\%$** between consecutive tunnels!