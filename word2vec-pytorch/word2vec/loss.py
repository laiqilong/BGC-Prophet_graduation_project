import torch.nn as nn
import torch.nn.functional as F

class SimoidBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, target, mask=None):
        # pred: (batch_size, max_len)
        # labels: (batch_size, max_len)
        # masks: (batch_size, max_len)
        out = F.binary_cross_entropy_with_logits(input=inputs, target=target, 
                                                 weight=mask, reduction='none')
        return out.mean(dim=1)
    