import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, reduction, gamma=1.0):
        """
        It always return "sum" of each loss
        """
        reduction = reduction.lower()
        assert reduction in ["sum", "mean"]
        super().__init__()
        self.gamma = gamma
        self.red = reduction

    def forward(self, netOutput, labels, alphas=None, state="mean"):
        """
        :param netOutput: [B, Classes]
        :param labels: [B]
        :param alphas: [B], the weight of each sample.
        """
        assert netOutput.sum().isnan().float() == 0, "Net Outputs have NaN value."
        b = labels.shape[0]
        bf = float(b)
        logprobs = F.log_softmax(netOutput, dim=-1)
        if alphas is None:
            alphas = torch.ones(size=[b, 1], device=netOutput.device)
        if len(alphas.shape) != 2:
            alphas = alphas[:, None]
        if len(labels.shape) != 2:
            labels = labels[:, None]
        logprobsTruth = torch.gather(logprobs, dim=-1, index=labels)
        with torch.no_grad():
            moderate = (1.0 - torch.exp(logprobsTruth)) ** self.gamma
        loss = -alphas * moderate * logprobsTruth
        if self.red == "sum" or state == "sum":
            return loss.sum()
        return loss.sum() / bf
