import argparse

import torch

import assignment.config as config
import assignment.libs.factory as factory


def parse_args():
    configs_available = config.list_available()

    parser = argparse.ArgumentParser(description="Compute mean and standard deviation of a dataset.")
    parser.add_argument("--config", help=f"Config name selected from {configs_available}", choices=configs_available, required=True)
    parser.add_argument("--split", help=f"Dataset split", default="training")
    args = parser.parse_args()

    return args.config, args.split


def compute_mean(dataloader):
    mean = 0.0
    for features, _ in dataloader:
        features = features.view(*features.shape[:2], -1)
        mean += torch.sum(torch.mean(features, dim=2), dim=0)
    mean = mean / len(dataloader.dataset)

    return mean


def compute_std(dataloader, mean):
    var = 0.0
    pixel_count = 0
    for features, _ in dataloader:
        features = features.view(*features.shape[:2], -1)
        var += torch.sum((features - mean[:, None]) ** 2, dim=(0, 2))
        pixel_count += features.nelement() / features.shape[1]
    std = torch.sqrt(var / pixel_count)

    return std


def compute_mean_and_std(name_config, split="training"):
    print(f"Computing mean and standard deviation ...")

    config.set_config_preset(name_config)

    dataset, dataloader = factory.create_dataset_and_dataloader(split=split)

    print("Dataset")
    print(dataset)

    mean = compute_mean(dataloader)
    std = compute_std(dataloader, mean)

    print(f"Mean: {mean}")
    print(f"Standard deviation: {std}")

    print(f"Computing mean and standard deviation finished")


def main():
    name_config, split = parse_args()
    compute_mean_and_std(name_config, split)


if __name__ == "__main__":
    main()
