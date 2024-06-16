import torch


class ELBOGaussian(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """A derivation: https://johfischer.com/2022/05/21/closed-form-solution-of-kullback-leibler-divergence-between-two-gaussians"""

        mu, log_var = output[1], output[2]
        loss = torch.mean(torch.sum(-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)), dim=1), dim=0)

        return loss
