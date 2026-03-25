"""Basic TurboQuant usage: quantize and dequantize vectors."""

import torch
from turboquant import TurboQuant

# Create quantizer: 128-dim vectors, 3 bits per coordinate
tq = TurboQuant(dim=128, bit_width=3, unbiased=True)

# Random vectors
x = torch.randn(10, 128)

# Quantize
output = tq.quantize(x)
print(f"MSE codes shape: {output.mse_codes.shape}")
print(f"QJL codes shape: {output.qjl_output.sign_bits.shape}")
print(f"Norms shape:     {output.mse_norms.shape}")

# Dequantize
x_hat = tq.dequantize(output)
print(f"Reconstructed shape: {x_hat.shape}")

# Measure distortion
mse = ((x - x_hat) ** 2).sum(dim=-1).mean()
rel_mse = ((x - x_hat) ** 2).sum(dim=-1) / (x ** 2).sum(dim=-1)
print(f"Mean MSE: {mse:.4f}")
print(f"Mean relative MSE: {rel_mse.mean():.4f}")
print(f"Compression ratio: {tq.compression_ratio():.1f}x")

# Inner product estimation
query = torch.randn(128)
true_ip = x @ query
estimated_ip = tq.compute_inner_product(query, output)
ip_error = (true_ip - estimated_ip).abs().mean()
print(f"Mean inner product error: {ip_error:.4f}")
