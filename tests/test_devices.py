import unittest
from unittest.mock import patch

from letsbuildmodels.devices import *

class TestGetDevice(unittest.TestCase):
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_cuda_available(self, mock_mps_available, mock_cuda_available):
        device = get_device()
        self.assertEqual(device, "cuda")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_available(self, mock_mps_available, mock_cuda_available):
        device = get_device()
        self.assertEqual(device, "mps")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_cpu_fallback(self, mock_mps_available, mock_cuda_available):
        device = get_device()
        self.assertEqual(device, "cpu")

class TestEmptyDevice(unittest.TestCase):
    @patch("torch.cuda.empty_cache")
    def test_empty_cuda_cache(self, mock_cuda_empty_cache):
        empty_device("cuda")
        mock_cuda_empty_cache.assert_called_once()

    @patch("torch.mps.empty_cache")
    def test_empty_mps_cache(self, mock_mps_empty_cache):
        empty_device("mps")
        mock_mps_empty_cache.assert_called_once()

    @patch("torch.cuda.empty_cache")
    @patch("torch.mps.empty_cache")
    def test_emptpy_cpu(self, mock_mps_empty_cache, mock_cuda_empty_cache):
        empty_device("cpus")
        mock_cuda_empty_cache.assert_not_called()
        mock_mps_empty_cache.assert_not_called()