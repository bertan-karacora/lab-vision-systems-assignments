import collections
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import assignment.config as config
import assignment.libs.utils_checkpoints as utils_checkpoints
import assignment.libs.utils_data as utils_data
import assignment.libs.utils_import as utils_import
import assignment.measurement


class Trainer:
    def __init__(self, name_exp, quiet=False):
        self.callbacks = None
        self.criterion = None
        self.dataloader_train = None
        self.dataloader_validate = None
        self.dataset_train = None
        self.dataset_validate = None
        self.device = None
        self.log = None
        self.measurers = None
        self.model = None
        self.name_exp = name_exp
        self.optimizer = None
        self.path_dir_exp = None
        self.scheduler = None
        self.quiet = quiet

        self._initialize()

    def _initialize(self):
        self.path_dir_exp = Path(config._PATH_DIR_EXPS) / self.name_exp

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = {
            "training": {
                "batches": {
                    "epoch": [],
                    "num_samples": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
                "epochs": {
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
            },
            "validation": {
                "batches": {
                    "epoch": [],
                    "num_samples": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
                "epochs": {
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
            },
        }

        self.setup_dataloaders()
        self.setup_model()
        self.setup_optimizer()
        self.setup_criterion()
        self.setup_measurers()

    def print(self, s):
        if not self.quiet:
            print(s)

    def setup_dataloaders(self):
        self.print("Setting up dataloaders...")

        self.dataset_train, self.dataloader_train = utils_data.create_dataset_and_dataloader(split="train")
        self.dataset_validate, self.dataloader_validate = utils_data.create_dataset_and_dataloader(split="validate")

        self.print("Train dataset")
        self.print(self.dataset_train)
        self.print("Validate dataset")
        self.print(self.dataset_validate)

        self.print("Setting up dataloaders finished")

    def setup_model(self):
        self.print("Setting up model...")

        class_model = utils_import.import_model(config.MODEL["name"])
        self.model = class_model(**config.MODEL["kwargs"]).eval()

        self.print("Model")
        self.print(self.model)
        # self.print(torchsummary.summary(model.cuda(), (labelset.size, DIMS_HIDDEN[0], dataset_train.data.shape[1:])))

        self.print("Setting up model finished")

    def setup_optimizer(self):
        self.print("Setting up optimizer...")

        class_optimizer = getattr(torch.optim, config.TRAINING["optimizer"]["name"])
        self.optimizer = class_optimizer(self.model.parameters(), **config.TRAINING["optimizer"]["kwargs"])

        if "scheduler" in config.TRAINING:
            class_scheduler = getattr(torch.optim.lr_scheduler, config.TRAINING["scheduler"]["name"])
            self.scheduler = class_scheduler(self.optimizer, **config.TRAINING["scheduler"]["kwargs"])

        self.print("Setting up optimizer finished")

    def setup_criterion(self):
        self.print("Setting up criterion...")

        class_criterion = getattr(torch.nn, config.CRITERION["name"])
        self.criterion = class_criterion(**config.CRITERION["kwargs"])

        self.print("Setting up criterion finished")

    def setup_measurers(self):
        self.print("Setting up measurers...")

        self.measurers = []
        for measurer in config.MEASURERS:
            class_measurer = getattr(assignment.measurement, measurer["name"])
            self.measurers += [class_measurer(**measurer["kwargs"])]

        self.print("Setting up measurers finished")

    @torch.no_grad()
    def validate_epoch(self, epoch):
        num_batches = len(self.dataloader_validate)
        progress_bar = tqdm(self.dataloader_validate, total=num_batches, disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            self.log["validation"]["batches"]["epoch"] += [epoch]
            self.log["validation"]["batches"]["num_samples"] += [len(targets)]

            features = features.to(self.device)
            targets = targets.to(self.device)
            output = self.model(features)

            loss = self.criterion(output, targets)
            # This is ugly and won't stay here. Generally, I will need to have multiple losses with weights and combine them.
            if "regularization" in config.CRITERION:
                if config.CRITERION["regularization"]["name"] == "L1":
                    params = torch.cat([param.view(-1) for param in self.model.parameters()])
                    loss += config.CRITERION["regularization"]["weight"] * torch.norm(params, 1.0)
            self.log["validation"]["batches"]["loss"] += [loss.item()]

            for measurer in self.measurers:
                metric = measurer(output, targets)
                self.log["validation"]["batches"]["metrics"][type(measurer).__name__] += [metric]

            if i % config.FREQUENCY_LOG == 0 and not self.quiet:
                progress_bar.set_description(f"Validating: Epoch {epoch:03d} | Batch {i:03d} | Loss {loss.item():.5f}")

        nums_samples = np.asarray(self.log["validation"]["batches"]["num_samples"][-num_batches:])

        losses = np.asarray(self.log["validation"]["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses * nums_samples) / len(self.dataset_validate)
        self.log["validation"]["epochs"]["loss"] += [loss_epoch]

        for name, metrics in self.log["validation"]["batches"]["metrics"].items():
            metrics_epoch = np.asarray(metrics[-num_batches:])
            metric_epoch = np.sum(metrics_epoch * nums_samples) / len(self.dataset_validate)
            self.log["validation"]["epochs"]["metrics"][name] += [metric_epoch]

    def train_epoch(self, epoch):
        num_batches = len(self.dataloader_train)
        progress_bar = tqdm(self.dataloader_train, total=num_batches, disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            self.log["training"]["batches"]["epoch"] += [epoch]
            self.log["training"]["batches"]["num_samples"] += [len(targets)]

            features = features.to(self.device)
            targets = targets.to(self.device)
            output = self.model(features)

            loss = self.criterion(output, targets)
            # This is ugly and won't stay here. Generally, I will need to have multiple losses with weights and combine them.
            if "regularization" in config.CRITERION:
                if config.CRITERION["regularization"]["name"] == "L1":
                    params = torch.cat([param.view(-1) for param in self.model.parameters()])
                    loss += config.CRITERION["regularization"]["weight"] * torch.norm(params, 1.0)

            self.log["training"]["batches"]["loss"] += [loss.item()]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for measurer in self.measurers:
                metric = measurer(output, targets)
                self.log["training"]["batches"]["metrics"][type(measurer).__name__] += [metric]

            if i % config.FREQUENCY_LOG == 0 and not self.quiet:
                progress_bar.set_description(f"Training: Epoch {epoch:03d} | Batch {i:03d} | Loss {loss.item():.5f}")

        nums_samples = np.asarray(self.log["training"]["batches"]["num_samples"][-num_batches:])

        losses = np.asarray(self.log["training"]["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses * nums_samples) / len(self.dataset_train)
        self.log["training"]["epochs"]["loss"] += [loss_epoch]

        for name, metrics in self.log["training"]["batches"]["metrics"].items():
            metrics_epoch = np.asarray(metrics[-num_batches:])
            metric_epoch = np.sum(metrics_epoch * nums_samples) / len(self.dataset_train)
            self.log["training"]["epochs"]["metrics"][name] += [metric_epoch]

    def loop(self, num_epochs, save_checkpoints=True):
        self.print("Looping...")

        self.model = self.model.to(self.device)

        self.validate_epoch(0)
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            self.train_epoch(epoch)

            self.model.eval()
            self.validate_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step(self.log["validation"]["epochs"]["loss"][-1])

            if save_checkpoints:
                utils_checkpoints.save(self, epoch, name="latest")
                if epoch % config.FREQUENCY_CHECKPOINT == 0 and epoch != 1:
                    utils_checkpoints.save(self, epoch)
                if epoch == num_epochs:
                    utils_checkpoints.save(self, epoch, name="final")

        self.print("Looping finished")
