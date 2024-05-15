import collections
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import assignment.config as config
import assignment.libs.utils_data as utils_data
import assignment.measurement


class Evaluator:
    def __init__(self, name_exp, model):
        self.callbacks = None
        self.criterion = None
        self.dataloader_test = None
        self.dataset_test = None
        self.device = None
        self.log = None
        self.measurers = None
        self.model = model
        self.name_exp = name_exp
        self.path_dir_exp = None

        self._initialize()

    def _initialize(self):
        self.path_dir_exp = Path(config._PATH_DIR_EXPS) / self.name_exp

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = {
            "batches": {
                "num_samples": [],
                "loss": [],
                "metrics": collections.defaultdict(list),
            },
            "total": {
                "loss": None,
                "metrics": {},
            },
        }

        self.setup_dataloader()
        self.setup_criterion()
        self.setup_measurers()

    def setup_dataloader(self):
        print("Setting up dataloader...")

        self.dataset_test, self.dataloader_test = utils_data.create_dataset_and_dataloader(split="test")

        print("Test dataset")
        print(self.dataset_test)

        print("Setting up dataloader finished")

    def setup_criterion(self):
        print("Setting up criterion...")

        class_criterion = getattr(torch.nn, config.CRITERION["name"])
        self.criterion = class_criterion(**config.CRITERION["kwargs"])

        print("Setting up criterion finished")

    def setup_measurers(self):
        print("Setting up measurers...")

        self.measurers = []
        for measurer in config.MEASURERS:
            class_measurer = getattr(assignment.measurement, measurer["name"])
            self.measurers += [class_measurer(**measurer["kwargs"])]

        print("Setting up measurers finished")

    @torch.no_grad()
    def evaluate(self, quiet=False):
        self.model = self.model.to(self.device)

        num_batches = len(self.dataloader_test)
        progress_bar = tqdm(self.dataloader_test, total=num_batches, disable=quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            self.log["batches"]["num_samples"] += [len(targets)]

            features = features.to(self.device)
            targets = targets.to(self.device)
            output = self.model(features)

            loss = self.criterion(output, targets)
            # This is ugly and won't stay here. Generally, I will need to have multiple losses with weights and combine them.
            if "regularization" in config.CRITERION:
                if config.CRITERION["regularization"]["name"] == "L1":
                    params = torch.cat([param.view(-1) for param in self.model.parameters()])
                    loss += config.CRITERION["regularization"]["weight"] * torch.norm(params, 1.0)
            self.log["batches"]["loss"] += [loss.item()]

            for measurer in self.measurers:
                metric = measurer(output, targets)
                self.log["batches"]["metrics"][type(measurer).__name__] += [metric]

            if i % config.FREQUENCY_LOG == 0 and not quiet:
                progress_bar.set_description(f"Validating: Batch {i:03d} | Loss {loss.item():.5f}")

        nums_samples = np.asarray(self.log["batches"]["num_samples"][-num_batches:])

        losses = np.asarray(self.log["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses * nums_samples) / len(self.dataset_test)
        self.log["total"]["loss"] = loss_epoch

        for name, metrics in self.log["batches"]["metrics"].items():
            metrics_total = np.asarray(metrics[-num_batches:])
            metric_total = np.sum(metrics_total * nums_samples) / len(self.dataset_test)
            self.log["total"]["metrics"][name] = metric_total
