import torch
import torch.nn as nn

torch.manual_seed(0)
experts, top_k = 4, 2
router = nn.Linear(8, experts, bias=False)
x = torch.randn(10, 8)

logits = router(x)
top_values, top_indices = logits.topk(top_k, dim=-1)

usage = torch.zeros(experts)
for expert_id in range(experts):
    positions, slots = torch.where(top_indices == expert_id)
    usage[expert_id] = positions.numel()
usage = usage / (10 * top_k)

balance = ((usage - 1.0 / experts) ** 2).mean() * experts
print("usage:", usage)
print("balance:", balance.item())
print("balance.requires_grad:", balance.requires_grad)