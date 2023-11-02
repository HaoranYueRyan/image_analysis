from scipy import ndimage
import numpy as np
from skimage.segmentation import clear_border
from skimage import measure,exposure, color
import matplotlib.pyplot as plt
import pathlib
from image_quantify.Aggreator import ImageAggregator
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


def aggregate_imgs(img_dict):
    """
    Aggregate images in well for specified channel and generate correction mask using the Aggregator Module
    :param channel: dictionary from self.exp_data.channels
    :return: flatfield correction mask for given channel
    """
    agg = ImageAggregator(15)
    for channel, img in img_dict.items():

        # image_array = generate_image(image, channel[1])
        agg.add_image(img)
    blurred_agg_img = agg.get_gaussian_image(12)
    return blurred_agg_img / blurred_agg_img.mean()
