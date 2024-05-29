import torch


class Accuracy:
    def __call__(self, output, targets):
        predictions = torch.argmax(output, dim=-1).flatten()

        if len(targets.shape) > 1:
            print(targets.shape)
            targets = torch.argmax(targets, dim=-1).flatten()
            print(targets.shape)

        num_positives_true = len(torch.where(predictions == targets)[0])
        num_total = len(targets)
        accuracy = num_positives_true / num_total

        return accuracy
