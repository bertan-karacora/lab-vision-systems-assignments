import sklearn.model_selection as model_selection
import torch

import assignment.config as config
import assignment.libs.factory as factory
import assignment.transforms.unnormalize as unnorm


def split_into_training_and_validation(dataset, ratio_validation_to_training):
    indices = list(range(len(dataset)))
    indices_training, indices_validation = model_selection.train_test_split(indices, test_size=ratio_validation_to_training)
    subset_training = torch.utils.data.Subset(dataset, indices_training)
    subset_validation = torch.utils.data.Subset(dataset, indices_validation)
    return subset_training, subset_validation


def sample(split, num_samples=None, use_unnormalize=False):
    num_samples = num_samples or config.DATA[split]["dataloader"]["kwargs"]["batch_size"]

    _, dataloader = factory.create_dataset_and_dataloader(split=split)

    features, target = sample_dataloader(dataloader, split=split, num_samples=num_samples, use_unnormalize=use_unnormalize)
    return features, target


def sample_dataloader(dataloader, split, num_samples=None, use_unnormalize=False):
    num_samples = num_samples or config.DATA[split]["dataloader"]["kwargs"]["batch_size"]

    features, target = next(iter(dataloader))
    features, target = features[:num_samples], target[:num_samples]

    if use_unnormalize:
        features = unnormalize(features, split=split)
        if "use_features_as_target" in config.DATA[split]["dataset"]["kwargs"] and config.DATA[split]["dataset"]["kwargs"]["use_features_as_target"]:
            target = unnormalize(target, split=split)

    return features, target


def sample_dataset(dataset, indices):
    lists = map(list, zip(*[dataset[i] for i in indices]))
    sample = torch.stack(lists)
    return sample


def unnormalize(input, split):
    output = unnorm.unnormalize(input, mean=config.DATA[split]["dataset"]["mean"], std=config.DATA[split]["dataset"]["std"])
    return output
