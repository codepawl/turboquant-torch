"""Test .to(device) works correctly across all classes."""

import pytest
import torch

from turboquant import TurboQuant, TurboQuantKVCache, TurboQuantMSE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
class TestDeviceTransfer:
    def test_turboquant_to_cuda(self):
        tq = TurboQuant(dim=64, bit_width=3, unbiased=True)
        tq = tq.to(torch.device("cuda"))
        x = torch.randn(5, 64, device="cuda")
        out = tq.quantize(x)
        x_hat = tq.dequantize(out)
        assert x_hat.device.type == "cuda"

    def test_mse_to_cuda(self):
        mse = TurboQuantMSE(dim=64, bit_width=2)
        mse = mse.to(torch.device("cuda"))
        x = torch.randn(5, 64, device="cuda")
        out = mse.quantize(x)
        x_hat = mse.dequantize(out)
        assert x_hat.device.type == "cuda"

    def test_kv_cache_to_cuda(self):
        cache = TurboQuantKVCache(head_dim=64, bit_width=3)
        cache = cache.to(torch.device("cuda"))
        keys = torch.randn(1, 2, 16, 64, device="cuda")
        values = torch.randn(1, 2, 16, 64, device="cuda")
        compressed = cache.compress(keys, values)
        k_hat = cache.decompress_keys(compressed)
        assert k_hat.device.type == "cuda"
