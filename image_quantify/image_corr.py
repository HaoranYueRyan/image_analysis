from scipy import ndimage
import numpy as np
from skimage.segmentation import clear_border
from skimage import measure,exposure, color
import matplotlib.pyplot as plt
import pathlib
def scaled(img: np.array, percentile: tuple[float, float] = (2, 98)) -> np.array:
    """Increase contrast by scaling image to exclude lowest and highest intensities"""
    percentiles = np.percentile(img, (percentile[0], percentile[1]))
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))

def normlize_img(image: np.array) -> np.array:
    mean = np.mean(image)
    std_dev = np.std(image)
    normalized_image = (image - mean) / std_dev
    return normalized_image


def filter_segmentation(mask: np.ndarray) -> np.ndarray:
    """
    removes border objects and filters large abd small objects from segmentation mask
    :param mask: unfiltered segmentation mask
    :return: filtered segmentation mask
    """
    cleared = clear_border(mask)
    labels, _ = ndimage.label(cleared)  # This ensures 'cleared' has integer labels
    sizes = np.bincount(labels.ravel())
    mask_sizes = sizes > 10
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[labels]
    return cells_cleaned.astype(int) * mask
def color_label(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Generate color labels for matplotlib to show segmentation
    :param mask: segmented mask
    :param img:
    :return: color labels for matplotlib
    """
    return color.label2rgb(mask, img, alpha=0.4, bg_label=0, kind='overlay')


def save_fig(path: pathlib, fig_id: str, tight_layout=True, fig_extension="pdf", resolution=300) -> None:
    """
    coherent saving of matplotlib figures as pdfs (default)
    :param path: path for saving
    :param fig_id: name of saved figure
    :param tight_layout: option, default True
    :param fig_extension: option, default pdf
    :param resolution: option, default 300dpi
    :return: None, saves Figure in poth
    """
    dest = path / f"{fig_id}.{fig_extension}"
    # print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(dest, format=fig_extension, dpi=resolution)
