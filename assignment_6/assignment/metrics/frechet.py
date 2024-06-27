import torch
import torchmetrics.image.fid as fid_tm

import assignment.transforms.unnormalize as unnormalize


class FrechetInceptionDistance(torch.nn.Module):
    def __init__(self, use_unnormalize=False, kwargs_unnormalize=None, **kwargs):
        super().__init__()

        self.kwargs_unnormalize = kwargs_unnormalize or {}
        self.metric_tm = fid_tm.FrechetInceptionDistance(**kwargs)
        self.use_unnormalize = use_unnormalize

    def forward(self, input, target):
        if self.use_unnormalize:
            input = unnormalize.unnormalize(input, **self.kwargs_unnormalize)
            target = unnormalize.unnormalize(target, **self.kwargs_unnormalize)

        self.metric_tm.update(target, real=True)
        self.metric_tm.update(input, real=False)
        output = self.metric_tm.compute()
        return output
