"""Approximate nearest neighbor search demo on synthetic data."""

import torch

from turboquant import TurboQuantIndex

# Parameters
dim = 128
n_db = 10000
n_queries = 10
k = 10
bit_width = 3

print(f"Database: {n_db} vectors, dim={dim}, {bit_width}-bit quantization")

torch.manual_seed(42)
database = torch.randn(n_db, dim)
queries = torch.randn(n_queries, dim)

# Build index (near-zero time, no training!)
index = TurboQuantIndex(dim=dim, bit_width=bit_width, metric="ip")
index.add(database)
print(f"Index built: {index.n_vectors} vectors, {index.memory_bytes() / 1024:.1f} KB")

scores, indices = index.search(queries, k=k)
print(f"Search results shape: scores={scores.shape}, indices={indices.shape}")

true_scores = queries @ database.t()
_, true_indices = torch.topk(true_scores, k, dim=-1)

recall_sum = 0
for i in range(n_queries):
    retrieved = set(indices[i].tolist())
    ground_truth = set(true_indices[i].tolist())
    recall_sum += len(retrieved & ground_truth) / k

recall = recall_sum / n_queries
print(f"Recall@{k}: {recall:.3f}")

original_bytes = n_db * dim * 4
compressed_bytes = index.memory_bytes()
print(
    f"Memory: {original_bytes / 1024:.1f} KB -> {compressed_bytes / 1024:.1f} KB "
    f"({original_bytes / compressed_bytes:.1f}x)"
)
