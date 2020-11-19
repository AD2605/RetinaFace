import torch
import torch.nn as nn

class focalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, ):
        super(focalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = torch.Tensor([alpha, 1-alpha])

    def forward(self, predictions, targets):
        if predictions.dim()>2:
            predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
            predictions = predictions.transpose(1, 2)
            predictions = predictions.contiguous().view(-1, predictions.size(2))

        targets = targets.view(-1, 1)
        log = torch.nn.functional.log_softmax(predictions)
        log = log.gather(1, targets)
        log = log.view(-1)
        pt = torch.tensor(log.data.exp())
        at = self.alpha.gather(0, targets.data.view(-1))
        log = log*at

        loss = -1*(1-pt)**self.gamma*log
        return loss.sum()
