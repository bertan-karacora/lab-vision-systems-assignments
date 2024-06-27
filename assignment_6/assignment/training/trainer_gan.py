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


class Trainer:
    def __init__(self, name_exp, quiet=False):
        self.criterion_discriminator_fake = None
        self.criterion_discriminator_real = None
        self.criterion_generator = None
        self.dataloader_training = None
        self.dataloader_validation = None
        self.dataset_training = None
        self.dataset_validation = None
        self.device = None
        self.log_discriminator = None
        self.log_generator = None
        self.measurers_training_discriminator = None
        self.measurers_training_generator = None
        self.measurers_validation_discriminator = None
        self.measurers_validation_generator = None
        self.model_discriminator = None
        self.model_generator = None
        self.name_exp = name_exp
        self.optimizer_discriminator = None
        self.optimizer_generator = None
        self.path_dir_exp = None
        self.scaler = None
        self.scheduler_discriminator = None
        self.scheduler_generator = None
        self.quiet = quiet
        self.writer_tensorboard = None

        self._init()

    def __str__(self):
        s = f"""Trainer for experiment {self.name_exp}
    Path: {self.path_dir_exp}
    Dataset (training): {self.dataset_training}
    Dataset (validation): {self.dataset_validation}
    Model (discriminator): {self.model_discriminator}
    Model (generator): {self.model_generator}
    Criterion (discriminator, fake): {self.criterion_discriminator_fake}
    Criterion (discriminator, real): {self.criterion_discriminator_real}
    Criterion (generator): {self.criterion_generator}
    Optimizer (discriminator): {self.optimizer_discriminator}
    Optimizer (generator): {self.optimizer_generator}
    Scheduler (discriminator): {self.scheduler_discriminator}
    Scheduler (generator): {self.scheduler_generator}
    Measurers (training): {self.measurers_training}
    Measurers (validation): {self.measurers_validation}"""
        return s

    def _init(self):
        self.path_dir_exp = Path(config._PATH_DIR_EXPS) / self.name_exp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer_tensorboard = SummaryWriter(self.path_dir_exp / "tensorboard")
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.TRAINING["use_amp"])
        self.log_discriminator = {
            "training": {
                "batches": {
                    "epoch": [],
                    "num_samples": [],
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
                "epochs": {
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
            },
            "validation": {
                "batches": {
                    "epoch": [],
                    "num_samples": [],
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
                "epochs": {
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
            },
        }
        self.log_generator = {
            "training": {
                "batches": {
                    "epoch": [],
                    "num_samples": [],
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
                "epochs": {
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
            },
            "validation": {
                "batches": {
                    "epoch": [],
                    "num_samples": [],
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
                "epochs": {
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
            },
        }
        self.dataset_training, self.dataloader_training = factory.create_dataset_and_dataloader(split="training")
        self.dataset_validation, self.dataloader_validation = factory.create_dataset_and_dataloader(split="validation")
        self.model_discriminator = factory.create_model(config.MODEL_DISCRIMINATOR)
        self.model_generator = factory.create_model(config.MODEL_GENERATOR)
        self.criterion_discriminator_fake = factory.create_criterion()
        self.criterion_discriminator_real = factory.create_criterion()
        self.criterion_generator = factory.create_criterion()
        self.optimizer_discriminator = factory.create_optimizer(self.model_discriminator.parameters())
        self.optimizer_generator = factory.create_optimizer(self.model_generator.parameters())
        if "scheduler" in config.TRAINING:
            self.scheduler_discriminator = factory.create_scheduler(self.optimizer_discriminator)
            self.scheduler_generator = factory.create_scheduler(self.optimizer_generator)
        self.measurers_training_discriminator = factory.create_measurers(config.MEASURERS_DISCRIMINATOR["training"])
        self.measurers_training_generator = factory.create_measurers(config.MEASURERS_DISCRIMINATOR["training"])
        self.measurers_validation_discriminator = factory.create_measurers(config.MEASURERS_GENERATOR["validation"])
        self.measurers_validation_generator = factory.create_measurers(config.MEASURERS_GENERATOR["validation"])

        self.print(self)
        self.print(torchsummary.summary(self.model_discriminator, [config.MODEL_DISCRIMINATOR["shape_input"]], verbose=0))
        self.print(torchsummary.summary(self.model_generator, [config.MODEL_GENERATOR["shape_input"]], verbose=0))

    def print(self, s):
        if not self.quiet:
            print(s)

    @torch.no_grad()
    def log_batch_discriminator(self, pass_loop, iteration, epoch, num_samples, loss, lr, output, targets):
        self.log_discriminator[pass_loop]["batches"]["epoch"] += [epoch]
        self.log_discriminator[pass_loop]["batches"]["num_samples"] += [num_samples]
        self.log_discriminator[pass_loop]["batches"]["loss"] += [loss.item()]
        self.log_discriminator[pass_loop]["batches"]["learning_rate"] += [lr]

        for measurer in getattr(self, f"measurers_{pass_loop}_discriminator"):
            name_metric = measurer.name_module if hasattr(measurer, "name_module") else type(measurer).__name__
            metric = measurer(output, targets)
            self.log_discriminator[pass_loop]["batches"]["metrics"][name_metric] += [metric.item()]
            self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Batches|{name_metric}", metric.item(), iteration)
        self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Batches|Loss", loss.item(), iteration)
        self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Batches|Learning rate", lr, iteration)

    @torch.no_grad()
    def log_batch_generator(self, pass_loop, iteration, epoch, num_samples, loss, lr, output, targets):
        self.log_generator[pass_loop]["batches"]["epoch"] += [epoch]
        self.log_generator[pass_loop]["batches"]["num_samples"] += [num_samples]
        self.log_generator[pass_loop]["batches"]["loss"] += [loss.item()]
        self.log_generator[pass_loop]["batches"]["learning_rate"] += [lr]

        for measurer in getattr(self, f"measurers_{pass_loop}_generator"):
            name_metric = measurer.name_module if hasattr(measurer, "name_module") else type(measurer).__name__
            metric = measurer(output, targets)
            self.log_generator[pass_loop]["batches"]["metrics"][name_metric] += [metric.item()]
            self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Batches|{name_metric}", metric.item(), iteration)
        self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Batches|Loss", loss.item(), iteration)
        self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Batches|Learning rate", lr, iteration)

    @torch.no_grad()
    def log_epoch_discriminator(self, pass_loop, epoch, num_samples, num_batches):
        nums_samples = np.asarray(self.log_discriminator[pass_loop]["batches"]["num_samples"][-num_batches:])

        losses = np.asarray(self.log_discriminator[pass_loop]["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses * nums_samples) / num_samples
        self.log_discriminator[pass_loop]["epochs"]["loss"] += [loss_epoch]

        lrs = np.asarray(self.log_discriminator[pass_loop]["batches"]["learning_rate"][-num_batches:])
        lr_epoch = np.sum(lrs * nums_samples) / num_samples
        self.log_discriminator[pass_loop]["epochs"]["learning_rate"] += [lr_epoch]

        for name, metrics in self.log_discriminator[pass_loop]["batches"]["metrics"].items():
            metrics_epoch = np.asarray(metrics[-num_batches:])
            metric_epoch = np.sum(metrics_epoch * nums_samples) / num_samples
            self.log_discriminator[pass_loop]["epochs"]["metrics"][name] += [metric_epoch]
            self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Epochs|{name}", metric_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Epochs|Loss", loss_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Epochs|Learning rate", lr_epoch, epoch)

    @torch.no_grad()
    def log_epoch_generator(self, pass_loop, epoch, num_samples, num_batches):
        nums_samples = np.asarray(self.log_generator[pass_loop]["batches"]["num_samples"][-num_batches:])

        losses = np.asarray(self.log_generator[pass_loop]["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses * nums_samples) / num_samples
        self.log_generator[pass_loop]["epochs"]["loss"] += [loss_epoch]

        lrs = np.asarray(self.log_generator[pass_loop]["batches"]["learning_rate"][-num_batches:])
        lr_epoch = np.sum(lrs * nums_samples) / num_samples
        self.log_generator[pass_loop]["epochs"]["learning_rate"] += [lr_epoch]

        for name, metrics in self.log_generator[pass_loop]["batches"]["metrics"].items():
            metrics_epoch = np.asarray(metrics[-num_batches:])
            metric_epoch = np.sum(metrics_epoch * nums_samples) / num_samples
            self.log_generator[pass_loop]["epochs"]["metrics"][name] += [metric_epoch]
            self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Epochs|{name}", metric_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Epochs|Loss", loss_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Epochs|Learning rate", lr_epoch, epoch)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader_validation, total=len(self.dataloader_validation), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = features.to(self.device)
            targets = targets.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler is not None):
                output = self.model(features)
                loss = self.criterion(output, targets)
            self.scaler.scale(loss)

            lr = self.optimizer_discriminator.param_groups[0]["lr"]
            self.log_batch("validation", len(self.dataloader_validation) * epoch + i, epoch, len(targets), loss, lr, output, targets)
            if i % config.LOGGING["tqdm"]["frequency"] == 0 and not self.quiet:
                progress_bar.set_description(f"Validating: Epoch {epoch:03d} | Batch {i:03d} | LR {lr:.6f} | Loss {loss.item():.5f}")

        self.log_epoch("validation", epoch, len(self.dataset_validation), len(self.dataloader_validation))

    def train_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader_training, total=len(self.dataloader_training), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = features.to(self.device)
            targets = targets.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler is not None):
                output = self.model(features)
                loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            self.criterion_g = lambda pred: F.binary_cross_entropy(pred, torch.ones(pred.shape[0], device=pred.device))
        self.criterion_d_real = lambda pred: F.binary_cross_entropy(pred, torch.ones(pred.shape[0], device=pred.device))
        self.criterion_d_fake = lambda pred: F.binary_cross_entropy(pred, torch.zeros(pred.shape[0], device=pred.device))


            lr = self.optimizer_discriminator.param_groups[0]["lr"]
            self.log_batch("training", len(self.dataloader_training) * epoch + i, epoch, len(targets), loss, lr, output, targets)
            if i % config.LOGGING["tqdm"]["frequency"] == 0 and not self.quiet:
                progress_bar.set_description(f"Training: Epoch {epoch:03d} | Batch {i:03d} | LR {lr:.6f} | Loss {loss.item():.5f}")

        self.log_epoch("training", epoch, len(self.dataset_training), len(self.dataloader_training))

    def loop(self, num_epochs, save_checkpoints=True):
        self.print("Looping ...")

        loss_best = float("inf")
        epoch_loss_best = 0
        self.model_discriminator = self.model_discriminator.to(self.device)
        self.model_generator = self.model_generator.to(self.device)
        self.criterion_discriminator_fake = self.criterion_discriminator_fake.to(self.device)
        self.criterion_discriminator_real = self.criterion_discriminator_real.to(self.device)
        self.criterion_generator = self.criterion_generator.to(self.device)
        for i in range(len(self.measurers_training_discriminator)):
            self.measurers_training_discriminator[i] = self.measurers_training_discriminator[i].to(self.device)
        for i in range(len(self.measurers_validation_discriminator)):
            self.measurers_validation_discriminator[i] = self.measurers_validation_discriminator[i].to(self.device)
        for i in range(len(self.measurers_training_generator)):
            self.measurers_training_generator[i] = self.measurers_training_generator[i].to(self.device)
        for i in range(len(self.measurers_validation_generator)):
            self.measurers_validation_generator[i] = self.measurers_validation_generator[i].to(self.device)

        self.validate_epoch(0)
        for epoch in range(1, num_epochs + 1):
            self.model_discriminator.train()
            self.model_generator.train()
            self.train_epoch(epoch)

            self.model_discriminator.eval()
            self.model_generator.eval()
            self.validate_epoch(epoch)

            loss_epoch = self.log_generator["validation"]["epochs"]["loss"][-1]
            if self.scheduler_discriminator is not None:
                self.scheduler_discriminator.step()
            if self.scheduler_generator is not None:
                self.scheduler_generator.step()

            if loss_epoch < loss_best:
                loss_best = loss_epoch
                epoch_loss_best = epoch
                if save_checkpoints:
                    utils_checkpoints.save(self, epoch, name="best")

            if save_checkpoints:
                utils_checkpoints.save(self, epoch, name="latest")
                if epoch % config.LOGGING["checkpoint"]["frequency"] == 0 and epoch != 1:
                    utils_checkpoints.save(self, epoch)
                if epoch == num_epochs:
                    utils_checkpoints.save(self, epoch, name="final")

            if "early_stopping" in config.TRAINING and epoch - epoch_loss_best > config.TRAINING["early_stopping"]["patience"]:
                self.print("Looping stopped early")
                break

            # TODO: Bad. But have no time left
            if "transfer" in config.MODEL and "epochs_freeze" in config.MODEL["transfer"] and epoch == config.MODEL["transfer"]["epochs_freeze"]:
                self.print("Training entire model now")
                for param in self.model.parameters():
                    param.requires_grad = True

        self.print("Looping finished")
