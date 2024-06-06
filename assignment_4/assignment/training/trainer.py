import collections
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchsummary
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
        self.writer = None

        self._init()

    def _init(self):
        self.path_dir_exp = Path(config._PATH_DIR_EXPS) / self.name_exp
        self.writer = SummaryWriter(self.path_dir_exp / "tensorboard")

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

    def log_batch(self, pass_loop, iteration, epoch, num_samples, loss, output, targets):
        self.log[pass_loop]["batches"]["epoch"] += [epoch]
        self.log[pass_loop]["batches"]["num_samples"] += [num_samples]
        self.log[pass_loop]["batches"]["loss"] += [loss]

        for measurer in self.measurers:
            name_metric = type(measurer).__name__
            metric = measurer(output, targets)
            self.log[pass_loop]["batches"]["metrics"][name_metric] += [metric]
            self.writer.add_scalar(f"{pass_loop}|Batches|{name_metric}", metric, iteration)
        self.writer.add_scalar(f"{pass_loop}|Batches|Loss", loss, iteration)

    def log_epoch(self, pass_loop, epoch, num_samples, num_batches):
        nums_samples = np.asarray(self.log[pass_loop]["batches"]["num_samples"][-num_batches:])

        losses = np.asarray(self.log[pass_loop]["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses * nums_samples) / num_samples
        self.log[pass_loop]["epochs"]["loss"] += [loss_epoch]

        for name, metrics in self.log[pass_loop]["batches"]["metrics"].items():
            metrics_epoch = np.asarray(metrics[-num_batches:])
            metric_epoch = np.sum(metrics_epoch * nums_samples) / num_samples
            self.log[pass_loop]["epochs"]["metrics"][name] += [metric_epoch]
            self.writer.add_scalar(f"{pass_loop}|Epochs|{name}", metric_epoch, epoch)
        self.writer.add_scalar(f"{pass_loop}|Epochs|Loss", loss_epoch, epoch)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader_validate, total=len(self.dataloader_validate), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = features.to(self.device)
            targets = targets.to(self.device)

            output = self.model(features)
            loss = self.criterion(output, targets)

            self.log_batch("validation", len(self.dataloader_validate) * epoch + i, epoch, len(targets), loss.item(), output, targets)
            lr = self.optimizer.param_groups[0]["lr"]
            if i % config.FREQUENCY_LOG == 0 and not self.quiet:
                progress_bar.set_description(f"Validating: Epoch {epoch:03d} | Batch {i:03d} | LR {lr:.6f} | Loss {loss.item():.5f}")

        self.log_epoch("validation", epoch, len(self.dataset_validate), len(self.dataloader_validate))

    def train_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader_train, total=len(self.dataloader_train), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = features.to(self.device)
            targets = targets.to(self.device)

            output = self.model(features)
            loss = self.criterion(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.log_batch("training", len(self.dataloader_train) * epoch + i, epoch, len(targets), loss.item(), output, targets)
            lr = self.optimizer.param_groups[0]["lr"]
            if i % config.FREQUENCY_LOG == 0 and not self.quiet:
                progress_bar.set_description(f"Training: Epoch {epoch:03d} | Batch {i:03d} | LR {lr:.6f} | Loss {loss.item():.5f}")

        self.log_epoch("training", epoch, len(self.dataset_train), len(self.dataloader_train))

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
