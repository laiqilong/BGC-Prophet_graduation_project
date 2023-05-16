import torch
import torch.nn as nn

class LSTMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, output, labels):
        # output: (batch_size, labels_num)
        return self.criterion(output, labels)


import torch.nn.functional as F

# class WeightedFocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"    
#     def __init__(self, alpha=.25, gamma=2):
#             super(WeightedFocalLoss, self).__init__()        
#             self.alpha = torch.tensor([alpha, 1-alpha]).cuda()        
#             self.gamma = gamma
            
#     def forward(self, inputs, targets):
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
#             targets = targets.type(torch.long)        
#             at = self.alpha.gather(0, targets.data.view(-1))        
#             pt = torch.exp(-BCE_loss).view(-1)
#             BCE_loss = BCE_loss.view(-1)        
#             F_loss = at*(1-pt)**self.gamma * BCE_loss        
#             return F_loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        BEC_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BEC_EXP) ** self.gamma * BCE

        return focal_loss