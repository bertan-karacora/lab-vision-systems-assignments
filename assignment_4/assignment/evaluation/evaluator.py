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
        self.criterion = None
        self.dataloader_test = None
        self.dataset_test = None
        self.device = None
        self.log = None
        self.measurers = None
        self.model = model
        self.name_exp = name_exp
        self.path_dir_exp = None

        self._init()

    def _init(self):
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

        self.init_dataloader()
        self.init_criterion()
        self.init_measurers()

    def init_dataloader(self):
        print("Initializing dataloader...")

        self.dataset_test, self.dataloader_test = utils_data.create_dataset_and_dataloader(split="test")

        print("Test dataset")
        print(self.dataset_test)

        print("Initializing dataloader finished")

    def init_criterion(self):
        print("Initializing criterion...")

        class_criterion = getattr(torch.nn, config.CRITERION["name"])
        self.criterion = class_criterion(**config.CRITERION["kwargs"])

        print("Initializing criterion finished")

    def init_measurers(self):
        print("Initializing measurers...")

        self.measurers = []
        for measurer in config.MEASURERS:
            class_measurer = getattr(assignment.measurement, measurer["name"])
            self.measurers += [class_measurer(**measurer["kwargs"])]

        print("Initializing measurers finished")

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