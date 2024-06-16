import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

import assignment.libs.utils_visualization as utils_visualization


def visualize_images(images, labels=None, indices=None, path_save=None):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subplot_image(image, i):
        ax = plt.gca()

        if labels is not None or indices is not None:
            title = ""
            if indices is not None:
                title += rf"#${indices[i]}$"
            if indices is not None and labels is not None:
                title += " | "
            if labels is not None:
                title += rf"label: {labels[i]}"
            ax.set_title(title, fontsize=9)
        ax.set_axis_off()

        image_vis = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        ax.imshow(image_vis, cmap="gray" if image.shape[-1] == 1 else None)

    images = images.numpy().transpose((0, 2, 3, 1))

    num_rows, num_cols = utils_visualization.get_dimensions(images, figsize=fig.get_size_inches())
    for i, image in enumerate(images):
        fig.add_subplot(num_rows, num_cols, i + 1)
        subplot_image(image, i)

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save)
    plt.show()


def visualize_kernels(kernels, channel=0):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subfigure_kernels(subfig, name, kernels_single):
        subfig.suptitle(name)

        def subplot_kernel(kernel):
            ax = plt.gca()
            ax.set_axis_off()
            kernel -= torch.min(kernel)
            kernel /= torch.max(kernel)
            ax.imshow(kernel, cmap="gray")

        # Assume same shape for all images
        aspect_images = kernels_single[0].shape[1] / kernels_single[0].shape[0]
        figsize = fig.get_size_inches()
        aspect_figure = len(kernels) * figsize[0] / figsize[1]

        num_subplots = len(kernels_single)
        num_cols = max(int(np.sqrt(num_subplots * aspect_figure / aspect_images)), 1)
        num_rows = np.ceil(num_subplots / num_cols).astype(int)
        for j, kernel in enumerate(kernels_single):
            subfig.add_subplot(num_rows, num_cols, j + 1)
            subplot_kernel(kernel)

    gs = gridspec.GridSpec(len(kernels), 1)
    for i, (name, kernels_single) in enumerate(kernels.items()):
        subfig = fig.add_subfigure(gs[i, :])
        subfigure_kernels(subfig, name, torch.squeeze(kernels_single[:, channel, :, :]))

    plt.show()


def visualize_featuremaps(featuremaps):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subfigure_featuremaps(subfig, name, featuremaps_single):
        subfig.suptitle(name)

        def subplot_featuremap(featuremap):
            ax = plt.gca()
            ax.set_axis_off()
            ax.imshow(featuremap)

        # Assume same shape for all images
        aspect_images = featuremaps_single[0].shape[1] / featuremaps_single[0].shape[0]
        figsize = fig.get_size_inches()
        aspect_figure = len(featuremaps) * figsize[0] / figsize[1]

        num_subplots = len(featuremaps_single)
        num_cols = max(int(np.sqrt(num_subplots * aspect_figure / aspect_images)), 1)
        num_rows = np.ceil(num_subplots / num_cols).astype(int)
        for j, featuremap in enumerate(featuremaps_single):
            subfig.add_subplot(num_rows, num_cols, j + 1)
            subplot_featuremap(featuremap)

    gs = gridspec.GridSpec(len(featuremaps), 1)
    for i, (name, featuremaps_single) in enumerate(featuremaps.items()):
        subfig = fig.add_subfigure(gs[i, :])
        subfigure_featuremaps(subfig, name, torch.squeeze(featuremaps_single))

    plt.show()
