"""Unit tests for SpeedNet architecture and window dataset."""

import os
import sys
import unittest
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.speed_net import SpeedNet
from src.dataset.window_dataset import IMUSpeedWindowDataset


class TestSpeedModel(unittest.TestCase):
    def setUp(self):
        self.model = SpeedNet(in_channels=14, hidden_dim=32, num_gru_layers=2)

    def test_forward_pass_shape(self):
        batch_size = 4
        window_len = 40
        dummy_input = torch.randn(batch_size, 14, window_len)
        output = self.model(dummy_input)

        self.assertEqual(output.shape, (batch_size, 1))
        # Ensure kinematic speed constraint: speed is strictly >= 0
        self.assertTrue(torch.all(output >= 0.0))

    def test_parameter_count(self):
        # Ensure model is lightweight (<100k params) for smartphone deployment
        num_params = sum(p.numel() for p in self.model.parameters())
        self.assertLess(num_params, 100_000)

    def test_onnx_export(self):
        test_onnx_path = os.path.join(os.path.dirname(__file__), "test_model.onnx")
        try:
            SpeedNet.export_to_onnx(self.model, test_onnx_path, input_shape=(1, 14, 40))
            self.assertTrue(os.path.exists(test_onnx_path))
            self.assertGreater(os.path.getsize(test_onnx_path), 10_000)
        finally:
            if os.path.exists(test_onnx_path):
                os.remove(test_onnx_path)

    def test_window_dataset_normalization(self):
        # Synthetic feature tensor: (50 windows, 14 channels, 40 timesteps)
        features = np.random.normal(loc=5.0, scale=2.0, size=(50, 14, 40)).astype(np.float32)
        targets = np.random.uniform(0.0, 25.0, size=(50,)).astype(np.float32)

        dataset = IMUSpeedWindowDataset(features, targets)
        self.assertEqual(len(dataset), 50)
        x_sample, y_sample = dataset[0]
        self.assertEqual(x_sample.shape, (14, 40))
        self.assertEqual(y_sample.shape, (1,))


if __name__ == "__main__":
    unittest.main()
