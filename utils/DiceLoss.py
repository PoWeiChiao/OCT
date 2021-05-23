import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        N = target.shape[0]
        smooth = 1

        input = torch.sigmoid(input) 

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss

class MultiClassDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, weights=None):
        C = target.shape[1]
        dice = DiceLoss()
        totalLoss = 0
        for i in range(1, C):
            target_one_category = torch.where(target == i, i, 0)
            print(target_one_category.size())
            print(input[:,1].size())
            loss = dice(input[:,i], target_one_category)
            if weights is not None:
                loss *= weights[i]
            totalLoss += loss
        return totalLoss