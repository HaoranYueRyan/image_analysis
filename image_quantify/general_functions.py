import skimage.io
import matplotlib.pyplot as plt
from image_corr import save_fig,scaled
import numpy as np
from image_quantify import Defaults
import pathlib

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


def gen_example(example_img,mask):
    # example_img = generate_random_image(well, channel)

    scaled_img = scaled(example_img[0])
    corr_img = example_img[0] / mask
    # corr_img=scaled_img
    bg_corr_img = corr_img - np.median(corr_img)
    # bg_corr_img[np.where(bg_corr_img <= 0.1)] = 0.1
    corr_scaled = scaled(bg_corr_img)
    # order all images for plotting
    img_list=[(scaled_img, 'original image'), (np.diagonal(scaled_img), 'diag. intensities'),
            (corr_scaled, 'corrected image'), (np.diagonal(bg_corr_img), 'diag. intensities'),
            (mask, 'flat_field MASK'),]


    fig, axes = plt.subplots(1, len(img_list), figsize=(25, len(img_list)))

    for ax, (data, title) in zip(axes, img_list):
        if len(data.shape) == 2:  # If the data is a 2D image
            ax.imshow(data, cmap='gray', aspect='auto')
        else:  # If the data is a 1D diagonal intensity
            ax.plot(data)
        ax.set_title(title)
        if len(data.shape) == 2:  # Only turn off axis for images, not for line plots
            ax.axis('off')

    # Save the figure to a PDF
    plt.tight_layout()
    plt.savefig(f'//Users/haoranyue/Desktop/Quantify_intensity_level/figures/fig_DAPI.pdf')
    plt.close(fig)


if __name__ == '__main__':
    ori_img = skimage.io.imread('/Users/haoranyue/Documents/P269_HumTORMP_TMA14_Scan1_component_data.tif')

    image_dict = {}
    # for count, well in enumerate(list(meta_data.plate_obj.listChildren())):
    for channel_num in range(ori_img.shape[0]):
        image_dict[Defaults.CHANNEL_NAME[int(channel_num)]] = np.array(ori_img[channel_num, :, :])

    norm_mask = aggregate_imgs(image_dict)
    img_list=gen_example(ori_img, norm_mask)

    fig, axes = plt.subplots(1,len(img_list), figsize=(25, len(img_list)))

    for ax, (data, title) in zip(axes, img_list):
        if len(data.shape) == 2:  # If the data is a 2D image
            ax.imshow(data, cmap='gray', aspect='auto')
        else:  # If the data is a 1D diagonal intensity
            ax.plot(data)
        ax.set_title(title)
        if len(data.shape) == 2:  # Only turn off axis for images, not for line plots
            ax.axis('off')

    # Save the figure to a PDF
    plt.tight_layout()
    plt.savefig('//Users/haoranyue/Desktop/Quantify_intensity_level/figures/output.pdf')
    plt.close(fig)

