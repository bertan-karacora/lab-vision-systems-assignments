import sklearn.model_selection as model_selection
import torch

import assignment.config as config
from assignment.transforms.unnormalize import Unnormalize


def split_into_training_and_validation(dataset, ratio_validation_to_training=0.8):
    indices = list(range(len(dataset)))
    indices_training, indices_validation = model_selection.train_test_split(indices, test_size=1.0 - ratio_validation_to_training)
    subset_training = torch.utils.data.Subset(dataset, indices_training)
    subset_validation = torch.utils.data.Subset(dataset, indices_validation)
    return subset_training, subset_validation


def sample(dataloader, num_samples=16, unnormalize=False):
    features, labels = next(iter(dataloader))
    features, labels = features[:num_samples], labels[:num_samples]

    if unnormalize:
        transform = Unnormalize(**config.VISUALIZATION["kwargs_unnormalize"])
        features = transform(features)

    return features, labels


def sample_dataset(dataset, indices):
    lists = map(list, zip(*[dataset[i] for i in indices]))
    sample = torch.stack(lists)
    return sample
