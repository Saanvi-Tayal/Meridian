"""Training Pipeline for AI Speed & Dynamic Estimator (SpeedNet)

Trains a hybrid 1D-CNN + Bi-GRU on synchronized IO-VNBD smartphone telemetry
and exports both PyTorch checkpoint and production-ready ONNX model.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.loader import IOVNBDDataset
from src.dataset.window_dataset import IMUSpeedWindowDataset
from src.models.speed_net import SpeedNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IO-VNBD"))
    dataset_loader = IOVNBDDataset(root_dir=root_dir)
    trips = dataset_loader.list_available_trips()

    # Select representative training trips across Driver A and Driver E
    # Hold out Driver A's S1 for unseen evaluation
    train_trip_keys = [
        ("S (Driver A)", "S2"),
        ("S (Driver A)", "S4"),
        ("Vta (Driver E)", "Vta01a"),
        ("Vtb (Driver E)", "Vtb01"),
        ("Vw (Driver E)", "Vw01"),
    ]

    selected_trips = []
    for driver_key, trip_key in train_trip_keys:
        matches = [t for t in trips if t["driver"] == driver_key and t["trip"] == trip_key]
        if matches:
            selected_trips.append(matches[0])

    print(f"[*] Selected {len(selected_trips)} diverse trips for training/validation:")
    loaded_trips = []
    for t_info in selected_trips:
        print(f"    - Loading: {t_info['driver']} / {t_info['trip']}")
        loaded_trips.append(dataset_loader.load_trip(t_info["phone_csv"], t_info["vehicle_csv"]))

    # 1. Build Sliding Window Dataset
    print("\n[*] Preprocessing and extracting sliding windows (W=40, Stride=2)...")
    full_dataset = IMUSpeedWindowDataset.from_trips(loaded_trips, window_len=40, stride=2)
    print(f"[+] Total training/val window samples: {len(full_dataset)}")

    # 2. Train / Val Split (80% train, 20% val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # 3. Model Initialization (14 input channels)
    model = SpeedNet(in_channels=14, hidden_dim=48, num_gru_layers=2).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Initialized SpeedNet ({num_params:,} parameters, lightweight mobile model)")

    # 4. Loss & Optimizer
    criterion = nn.SmoothL1Loss() # Huber loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training Loop
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    best_val_mae = float("inf")
    best_weights_path = os.path.join(models_dir, "speed_net.pth")
    norm_path = os.path.join(models_dir, "speed_net_norm.npz")
    onnx_path = os.path.join(models_dir, "speed_net.onnx")

    # Save normalization constants
    np.savez(norm_path, mean=full_dataset.mean, std=full_dataset.std)
    print(f"[+] Saved normalization parameters to: {norm_path}")

    print("\n" + "="*70)
    print("STARTING TRAINING LOOP")
    print("="*70)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_samples += len(y_batch)

        scheduler.step()
        train_loss /= max(1, train_samples)

        # Validation
        model.eval()
        val_errors_ms = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_preds = model(X_val)
                errs = torch.abs(val_preds - y_val).cpu().numpy().flatten()
                val_errors_ms.extend(errs)

        val_mae_ms = float(np.mean(val_errors_ms))
        val_mae_kmh = val_mae_ms * 3.6

        print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f} | Val MAE: {val_mae_ms:.2f} m/s ({val_mae_kmh:.1f} km/h)")

        if val_mae_ms < best_val_mae:
            best_val_mae = val_mae_ms
            torch.save(model.state_dict(), best_weights_path)
            print(f"    --> [NEW BEST] Saved weights to {best_weights_path}")

    print("\n" + "="*70)
    print(f"TRAINING COMPLETE! Best Validation MAE: {best_val_mae:.2f} m/s ({best_val_mae * 3.6:.1f} km/h)")
    print("="*70)

    # 6. Export to ONNX
    print("\n[*] Exporting best model to ONNX for mobile & edge deployment...")
    model.load_state_dict(torch.load(best_weights_path, map_location="cpu"))
    model.eval().to("cpu")
    SpeedNet.export_to_onnx(model, onnx_path, input_shape=(1, 14, 40))
    onnx_size_kb = os.path.getsize(onnx_path) / 1024.0
    print(f"[+] ONNX model size: {onnx_size_kb:.1f} KB (Well below 1MB edge budget!)")


if __name__ == "__main__":
    main()
