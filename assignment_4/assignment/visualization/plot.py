import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

import assignment.libs.utils_visualization as utils_visualization


def plot_loss(log, path_save=None):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subplot_loss(use_logscale=False):
        ax = plt.gca()

        ax.set_title(f"Training progress ({'logscale' if use_logscale else 'linearscale'})", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Cross-entropy loss", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(alpha=0.4)
        if use_logscale:
            ax.set_yscale("log")

        epochs_train = np.asarray(log["training"]["batches"]["epoch"])
        nums_samples_train = np.asarray(log["training"]["batches"]["num_samples"])
        epochs_continuous = utils_visualization.create_epochs_continuous(epochs_train, nums_samples_train)

        loss_train = np.asarray(log["training"]["batches"]["loss"])

        num_iterations_per_epoch = len(epochs_train[epochs_train == epochs_train[0]])
        losses_train_smoothed = utils_visualization.smooth(loss_train, k=int(0.5 * num_iterations_per_epoch))

        ax.plot(epochs_continuous, loss_train, alpha=0.5)
        ax.plot(epochs_continuous, losses_train_smoothed, label="Loss (training)")
        ax.plot(log["validation"]["epochs"]["loss"], label="Loss (validation)")

        ax.legend(fontsize=9)

    fig.add_subplot(3, 1, 1)
    subplot_loss()

    fig.add_subplot(3, 1, 2)
    subplot_loss(use_logscale=True)

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save)
    plt.show()


def plot_metrics(log, path_save=None):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subplot_metrics(use_logscale=False):
        ax = plt.gca()

        ax.set_title(f"Training progress ({'logscale' if use_logscale else 'linearscale'})", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Metric", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(alpha=0.4)
        if use_logscale:
            ax.set_yscale("log")

        epochs_train = np.asarray(log["training"]["batches"]["epoch"])
        nums_samples_train = np.asarray(log["training"]["batches"]["num_samples"])
        epochs_continuous = utils_visualization.create_epochs_continuous(epochs_train, nums_samples_train)

        for name, metrics in log["training"]["batches"]["metrics"].items():
            metrics_train = np.asarray(metrics)

            num_iterations_per_epoch = len(epochs_train[epochs_train == epochs_train[0]])
            metrics_train_smoothed = utils_visualization.smooth(metrics_train, k=int(0.5 * num_iterations_per_epoch))

            ax.plot(epochs_continuous, metrics_train, alpha=0.5)
            ax.plot(epochs_continuous, metrics_train_smoothed, label=f"{name.capitalize()} (training)")

        for name, metrics in log["validation"]["epochs"]["metrics"].items():
            metrics_validate = np.asarray(metrics)
            ax.plot(metrics_validate, label=f"{name.capitalize()} (validation)")

        ax.legend(fontsize=9)

    fig.add_subplot(3, 1, 1)
    subplot_metrics()

    fig.add_subplot(3, 1, 2)
    subplot_metrics(use_logscale=True)

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save)
    plt.show()


def plot_confusion(confusion, labelset):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subplot_confusion():
        ax = plt.gca()

        confusion_normalized = confusion / np.sum(confusion, axis=1)
        df_confusion = pd.DataFrame(confusion_normalized, index=labelset, columns=labelset)

        sbn.heatmap(df_confusion, annot=True)

        ax.set_title("Confusion matrix", fontsize=9)
        ax.set_xlabel("Target", fontsize=9)
        ax.set_ylabel("Prediction", fontsize=9)
        ax.tick_params(bottom=False, left=False)

    fig.add_subplot(3, 1, 1)
    subplot_confusion()

    plt.tight_layout()
    plt.show()


# def plot_gradient_stats(iterations_train, stats, num_samples_train):
#     din_a4 = np.array([210, 297]) / 25.4
#     fig = plt.figure(figsize=din_a4)

#     def subplot_gradient_stat(name, stat):
#         ax = plt.gca()

#         ax.set_title(f"Gradient stat: {name.capitalize()}", fontsize=9)
#         ax.set_xlabel("Epoch", fontsize=9)
#         ax.set_ylabel(f"{name.capitalize()}", fontsize=9)
#         ax.tick_params(axis="both", which="major", labelsize=9)
#         ax.tick_params(axis="both", which="minor", labelsize=8)
#         ax.grid(alpha=0.4)

#         epochs_iterations_train = iterations2epochs(iterations_train, BATCHSIZE, num_samples_train)
#         for name_parameter, parameter in stat.items():
#             ax.plot(epochs_iterations_train, parameter, alpha=0.6, label=f"{name_parameter}")

#         ax.legend(fontsize=9)

#     for i, (name, stat) in enumerate(stats.items()):
#         fig.add_subplot(len(stats), 1, i + 1)
#         subplot_gradient_stat(name, stat)

#     plt.tight_layout()
#     plt.show()
