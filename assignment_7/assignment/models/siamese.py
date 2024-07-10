import torch
import torchvision as tv

from assignment.models.modules import Normalize


class TriNetAdapted(torch.nn.Module):
    def __init__(self, num_channels_latent, weights_mobilenet=None):
        super().__init__()

        self.num_channels_latent = num_channels_latent
        self.weights_mobilenet = weights_mobilenet

        self._init()

    def _init(self):
        mobilenet = tv.models.mobilenet_v3_small(weights=self.weights_mobilenet)
        self.backbone = torch.nn.Sequential(
            mobilenet.features,
            mobilenet.avgpool,
        )

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(576, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_channels_latent),
            Normalize(p_norm=2, dim=-1),
        )

    def forward_single(self, input):
        output = self.backbone(input)
        output = self.head(output)
        return output

    def forward(self, input):
        anchor = input["anchor"]
        positive = input["positive"]
        negative = input["negative"]

        inputs = torch.cat([anchor, positive, negative], dim=0)
        outputs = self.forward_single(inputs)
        output_anchor, output_positive, output_negative = torch.chunk(outputs, 3, dim=0)

        output = dict(anchor=output_anchor, positive=output_positive, negative=output_negative)
        return output
