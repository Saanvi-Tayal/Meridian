"""AI Speed & Dynamic Estimator Model

A lightweight, mobile-optimized 1D-CNN + Bi-GRU network that infers vehicle forward
velocity directly from vehicle-frame smartphone IMU telemetry.
"""

import os
from typing import Tuple
import torch
import torch.nn as nn


class SpeedNet(nn.Module):
    """
    Hybrid Convolutional-Recurrent Neural Network for IMU Speed Regression.
    Input: (B, 14, W) -> [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, acc_mag, gyro_mag, jerk_x, jerk_y, jerk_z, centripetal, kinetic_power, is_stationary]
    Output: (B, 1) -> Estimated forward speed v_fwd >= 0 (m/s)
    """

    def __init__(self, in_channels: int = 14, hidden_dim: int = 48, num_gru_layers: int = 2):
        super().__init__()
        self.in_channels = in_channels

        # 1. 1D-CNN Feature Extractor (Extracts local motion patterns & frequency signatures)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2), # Reduces temporal dimension by 2x
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # 2. Bidirectional GRU (Models vehicle inertia, acceleration curves, coasting)
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_gru_layers > 1 else 0.0,
        )

        # 3. Kinematic Speed Head (Predicts forward velocity in m/s)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(48, 1),
            nn.ReLU(), # Kinematic constraint: vehicle forward speed is non-negative (>= 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Shape (B, C_in, W) where C_in=11, W=window_length
        returns: (B, 1) scalar speed in m/s
        """
        # CNN: (B, 11, W) -> (B, 64, W//2)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        # Transpose for GRU: (B, 64, T) -> (B, T, 64)
        h = h.transpose(1, 2)

        # GRU: (B, T, 64) -> (B, T, hidden_dim * 2)
        gru_out, _ = self.gru(h)

        # Mean pooling across the temporal sequence to combine all window cues
        pooled = torch.mean(gru_out, dim=1)

        # Regression Head: (B, hidden_dim * 2) -> (B, 1)
        speed = self.head(pooled)
        return speed

    @classmethod
    def export_to_onnx(
        cls,
        model: "SpeedNet",
        output_path: str,
        input_shape: Tuple[int, int, int] = (1, 11, 40)
    ):
        """Exports the PyTorch model to standard ONNX format for on-device deployment."""
        model.eval()
        dummy_input = torch.randn(*input_shape, dtype=torch.float32)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["imu_window"],
            output_names=["predicted_speed_ms"],
            dynamic_axes={
                "imu_window": {0: "batch_size"},
                "predicted_speed_ms": {0: "batch_size"}
            },
            dynamo=False
        )
        print(f"[+] Model successfully exported to ONNX: {output_path}")
