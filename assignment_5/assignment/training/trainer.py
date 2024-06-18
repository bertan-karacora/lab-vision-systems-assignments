import collections
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchsummary
from tqdm import tqdm

import assignment.config as config
import assignment.libs.factory as factory
import assignment.libs.utils_checkpoints as utils_checkpoints
import assignment.metrics


class Trainer:
    def __init__(self, name_exp, quiet=False):
        self.criterion = None
        self.dataloader_training = None
        self.dataloader_validation = None
        self.dataset_training = None
        self.dataset_validation = None
        self.device = None
        self.log = None
        self.measurers = None
        self.model = None
        self.name_exp = name_exp
        self.optimizer = None
        self.path_dir_exp = None
        self.scheduler = None
        self.quiet = quiet
        self.writer_tensorboard = None

        self._init()

    def __str__(self):
        s = f"""Trainer for experiment {self.name_exp}
    Path: {self.path_dir_exp}
    Dataset (training): {self.dataset_training}
    Dataset (validation): {self.dataset_validation}
    Model: {self.model}
    Criterion: {self.criterion}
    Optimizer: {self.optimizer}
    Measurers: {self.measurers}"""
        return s

    def _init(self):
        self.path_dir_exp = Path(config._PATH_DIR_EXPS) / self.name_exp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer_tensorboard = SummaryWriter(self.path_dir_exp / "tensorboard")
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

        self.dataset_training, self.dataloader_training = factory.create_dataset_and_dataloader(split="training")
        self.dataset_validation, self.dataloader_validation = factory.create_dataset_and_dataloader(split="validation")
        self.model = utils_checkpoints.load_model(self.path_dir_exp / "checkpoints" / f"{self.name_checkpoint}.pth")
        self.measurers = factory.create_measurers()

        self.setup_model()
        self.setup_optimizer()
        self.setup_criterion()
        self.setup_measurers()

    def print(self, s):
        if not self.quiet:
            print(s)

    def setup_model(self):
        class_model = utils_import.import_model(config.MODEL["name"])
        self.model = class_model(**config.MODEL["kwargs"]).eval()

        if "transfer" in config.MODEL:
            # Won't stay like this. Assume we start with epoch 0 for now.
            if "epochs_freeze" in config.MODEL["transfer"] and config.MODEL["transfer"]["epochs_freeze"] > 0:
                print("Freezing body params")
                for param in self.model.parameters():
                    param.requires_grad = False

            for dict_layer in config.MODEL["transfer"]["layers"]:
                dict_model_layer = dict_layer["model"]
                class_model_layer = utils_import.import_model(dict_model_layer["name"])
                model_layer = class_model_layer(**dict_model_layer["kwargs"]).eval()

                setattr(self.model, dict_layer["name"], model_layer)

        self.print("Model")
        self.print(self.model)
        self.print(torchsummary.summary(self.model, [config.MODEL["shape_input"]], verbose=0))

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

        # TODO: combine multiple losses with weights. Adapt loss to losses in log etc.
        class_criterion = getattr(torch.nn, config.CRITERION["name"])
        self.criterion = class_criterion(**config.CRITERION["kwargs"])

        self.print("Setting up criterion finished")

    def setup_measurers(self):
        self.print("Setting up measurers...")

        self.measurers = []
        for measurer in config.MEASURERS:
            class_measurer = getattr(assignment.metrics, measurer["name"])
            self.measurers += [class_measurer(**measurer["kwargs"])]

        self.print("Setting up measurers finished")

    def log_batch(self, pass_loop, iteration, epoch, num_samples, loss, output, targets):
        self.log[pass_loop]["batches"]["epoch"] += [epoch]
        self.log[pass_loop]["batches"]["num_samples"] += [num_samples]
        self.log[pass_loop]["batches"]["loss"] += [loss]

        for measurer in self.measurers:
            name_metric = type(measurer).__name__
            metric = measurer(output, targets)
            self.log[pass_loop]["batches"]["metrics"][name_metric] += [metric]
            self.writer_tensorboard.add_scalar(f"{pass_loop}|Batches|{name_metric}", metric, iteration)
        self.writer_tensorboard.add_scalar(f"{pass_loop}|Batches|Loss", loss, iteration)

    def log_epoch(self, pass_loop, epoch, num_samples, num_batches):
        nums_samples = np.asarray(self.log[pass_loop]["batches"]["num_samples"][-num_batches:])

        losses = np.asarray(self.log[pass_loop]["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses * nums_samples) / num_samples
        self.log[pass_loop]["epochs"]["loss"] += [loss_epoch]

        for name, metrics in self.log[pass_loop]["batches"]["metrics"].items():
            metrics_epoch = np.asarray(metrics[-num_batches:])
            metric_epoch = np.sum(metrics_epoch * nums_samples) / num_samples
            self.log[pass_loop]["epochs"]["metrics"][name] += [metric_epoch]
            self.writer_tensorboard.add_scalar(f"{pass_loop}|Epochs|{name}", metric_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"{pass_loop}|Epochs|Loss", loss_epoch, epoch)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader_validation, total=len(self.dataloader_validation), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = features.to(self.device)
            targets = targets.to(self.device)

            output = self.model(features)
            loss = self.criterion(output, targets)

            self.log_batch("validation", len(self.dataloader_validation) * epoch + i, epoch, len(targets), loss.item(), output, targets)
            lr = self.optimizer.param_groups[0]["lr"]
            if i % config.FREQUENCY_LOG == 0 and not self.quiet:
                progress_bar.set_description(f"Validating: Epoch {epoch:03d} | Batch {i:03d} | LR {lr:.6f} | Loss {loss.item():.5f}")

        self.log_epoch("validation", epoch, len(self.dataset_validation), len(self.dataloader_validation))

    def train_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader_training, total=len(self.dataloader_training), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = features.to(self.device)
            targets = targets.to(self.device)

            output = self.model(features)
            loss = self.criterion(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.log_batch("training", len(self.dataloader_training) * epoch + i, epoch, len(targets), loss.item(), output, targets)
            lr = self.optimizer.param_groups[0]["lr"]
            if i % config.FREQUENCY_LOG == 0 and not self.quiet:
                progress_bar.set_description(f"Training: Epoch {epoch:03d} | Batch {i:03d} | LR {lr:.6f} | Loss {loss.item():.5f}")

        self.log_epoch("training", epoch, len(self.dataset_training), len(self.dataloader_training))

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

            # Bad. But have no time left
            if "transfer" in config.MODEL and "epochs_freeze" in config.MODEL["transfer"] and epoch == config.MODEL["transfer"]["epochs_freeze"]:
                print("Training entire model now")
                for param in self.model.parameters():
                    param.requires_grad = True

            if save_checkpoints:
                utils_checkpoints.save(self, epoch, name="latest")
                if epoch % config.FREQUENCY_CHECKPOINT == 0 and epoch != 1:
                    utils_checkpoints.save(self, epoch)
                if epoch == num_epochs:
                    utils_checkpoints.save(self, epoch, name="final")

        self.print("Looping finished")
