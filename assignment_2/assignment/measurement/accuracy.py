import torch


class Accuracy:
    def __call__(self, output, targets):
        predictions = torch.argmax(output, dim=-1)

        num_positives_true = len(torch.where(predictions == targets)[0])
        num_total = len(targets)
        accuracy = num_positives_true / num_total

        return accuracy
