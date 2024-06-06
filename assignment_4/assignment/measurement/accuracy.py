import torch


class Accuracy:
    def __call__(self, outputs, targets):
        predictions = torch.argmax(outputs, dim=-1).flatten()
        labels = torch.argmax(outputs, dim=-1).flatten() if len(targets.shape) > 1 else targets

        num_positives_true = len(torch.where(predictions == labels)[0])
        num_total = len(labels)
        accuracy = num_positives_true / num_total
        return accuracy
