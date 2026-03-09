"""Gate script for Phase 3 Task 3.1 — metrics module verification."""
import torch
from src.utils.metrics import ChangeDetectionMetrics

metrics = ChangeDetectionMetrics()

# Perfect prediction
pred = torch.tensor([1.0, 1.0, 0.0, 0.0])
target = torch.tensor([1, 1, 0, 0])
metrics.update(pred, target)
result = metrics.compute()
assert abs(result["f1"] - 1.0) < 1e-4, f"Perfect pred should give F1=1, got {result['f1']}"
metrics.reset()

# Imperfect prediction
pred = torch.tensor([0.9, 0.9, 0.9, 0.1])
target = torch.tensor([1, 0, 1, 0])
metrics.update(pred, target)
result = metrics.compute()
assert 0 < result["f1"] < 1.0, f"Imperfect pred F1 should be in (0,1), got {result['f1']}"

# Batched accumulation: same as above but across 2 update calls
metrics.reset()
pred_batch1 = torch.tensor([0.9, 0.9])
target_batch1 = torch.tensor([1, 0])
pred_batch2 = torch.tensor([0.9, 0.1])
target_batch2 = torch.tensor([1, 0])
metrics.update(pred_batch1, target_batch1)
metrics.update(pred_batch2, target_batch2)
result2 = metrics.compute()
# Should equal result (same data, just split across two updates)
assert abs(result2["f1"] - result["f1"]) < 1e-4, (
    f"Batched accumulation mismatch: {result2['f1']} vs {result['f1']}"
)

print("===================================")
print("  Task 3.1: Metrics verification")
print("===================================")
for k, v in result.items():
    print(f"  {k}: {v:.4f}")
print("  Batched accumulation: VERIFIED")
print("  STATUS: METRICS MODULE OK")
print("===================================")
