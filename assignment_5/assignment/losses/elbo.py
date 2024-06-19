import torch


class ELBOGaussian(torch.nn.Module):
    """Evidence lower bound assuming a Gaussian distribution.
    A derivation of the closed form solution: https://johfischer.com/2022/05/21/closed-form-solution-of-kullback-leibler-divergence-between-two-gaussians"""

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        mean, log_var = input["mean"], input["log_var"]

        loss = torch.mean(torch.sum(-0.5 * (1 + log_var - mean**2 - torch.exp(log_var)), dim=1), dim=0)
        return loss


# class TestLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input, target):
#         mean, log_var = input["mean"], input["log_var"]

#         recons_loss = torch.nn.functional.mse_loss(input["prediction"], target)
#         kld = (-0.5 * (1 + log_var - mean**2 - log_var.exp()).sum(dim=1)).mean(dim=0)
#         loss = recons_loss + 1.0e-3 * kld

#         # global COUNTER
#         # COUNTER += 1
#         # if COUNTER % 1000 == 0:
#         #     print("MSE", recons_loss)
#         #     print("KLD:", kld)

#         return loss
