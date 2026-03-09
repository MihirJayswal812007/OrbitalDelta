"""Phase 2 quick gate test."""
import torch
from src.models.siamese_unet import SiameseUNet
from src.models.losses import BCEDiceLoss

model = SiameseUNet("resnet18", pretrained=False, dropout=0.0)
a = torch.randn(1, 3, 128, 128)
b = torch.randn(1, 3, 128, 128)

out = model(a, b)
assert out.shape == (1, 1, 128, 128), f"Bad shape: {out.shape}"
assert 0 <= out.min().item() <= out.max().item() <= 1.0
assert not hasattr(model, "encoder_b"), "FAIL: separate encoder_b — weights not shared"

p = model.count_parameters()
assert p["total_M"] < 20, f"Too many params: {p['total_M']}M"

loss_fn = BCEDiceLoss()
target = (torch.rand(1, 1, 128, 128) > 0.8).float()
loss = loss_fn(out, target)
assert not torch.isnan(loss), "Loss is NaN!"
loss.backward()
grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
assert grads_ok, "Missing gradients!"

print("===================================")
print("  PHASE 2 GATE PASSED")
print(f"  Output: {list(out.shape)}")
print(f"  Params: {p['total_M']}M")
print(f"  Loss:   {loss.item():.4f}")
print(f"  Grads:  {grads_ok}")
print("  Weight sharing: single encoder verified")
print("===================================")
