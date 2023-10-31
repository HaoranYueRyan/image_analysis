import skimage.io
import torch
from image_quantify import Defaults
from image_quantify.image_corr import filter_segmentation
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cellpose import models
from skimage import measure, io
from image_quantify.image_corr import scaled,color_label,save_fig
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from cellpose import models


class Image:
    """
    Generates the corrected images and segmentation masks.
    Stores corrected images as dict, and n_mask, c_mask, and cyto_mask arrays.
    """

    def __init__(self, img_dict, exp_paths):
        self.img_dict = img_dict
        self._paths = exp_paths

        self.masks = {}
        for channel in ['DAPI', 'CD8', 'CD66b', 'CD68', 'CD16']:
            mask_path = os.path.join(self._paths.Mask, f'segmented_mask_{channel}.tif')
            if os.path.exists(mask_path):
                self.masks[channel] = skimage.io.imread(mask_path)  # load existing mask
            else:
                if channel == 'DAPI':
                    self.masks[channel] = self._n_segmentation()
                else:
                    self.masks[channel] = self._c_segmentation(self.img_dict[channel])
                skimage.io.imsave(os.path.join(self._paths.Mask, f'segmented_mask_{channel}.tif'), self.masks[channel])

    def _n_segmentation(self):
        model = models.Cellpose(gpu=True, model_type='nuclei')
        n_channels = [0, 0]
        n_mask_array, _, _, _ = model.eval(scaled(self.img_dict['DAPI']), diameter=12, channels=n_channels)
        return filter_segmentation(n_mask_array)

    def _c_segmentation(self, select_img):
        model = models.Cellpose(gpu=True, model_type='cyto')
        c_channels = [2, 1]
        comb_image = np.dstack([scaled(self.img_dict['DAPI']), scaled(select_img)])
        c_masks_array, _, _, _ = model.eval(comb_image, channels=c_channels, diameter=12)
        return filter_segmentation(c_masks_array)

    def segmentation_figure(self):
        channels = ['DAPI', 'CD8', 'CD66b', 'CD68', 'CD16']

        for channel in channels:
            mask_path = os.path.join(self._paths.example_img, f'corr_img_{channel}.tif')
            fig_path = os.path.join(self._paths.quality_ctr, f'{channel}_quality_segmentation_check.pdf')

            if not os.path.exists(mask_path):
                img_scaled = scaled(self.img_dict[channel])
                skimage.io.imsave(mask_path, img_scaled)

            if os.path.exists(fig_path):  # If the figure already exists, don't create it again
                continue

            # Only for the current channel
            original_img = self.img_dict[channel]
            color_labels = color_label(self.masks[channel], img_scaled)

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

            ax[0].axis('off')
            ax[0].imshow(original_img, cmap='gray')
            ax[0].title.set_text(f"{channel} image")

            ax[1].axis('off')
            ax[1].imshow(color_labels, cmap='gray')
            ax[1].title.set_text(f"{channel} segmentation")

            save_fig(self._paths.quality_ctr, f'{channel}_quality_segmentation_check')
            plt.close(fig)

class ImageProperties:
    """
    Extracts feature measurements from segmented nuclei, cells and cytoplasm
    and generates combined data frames.
    """

    def __init__(self,  image_obj, featurelist=None):
        if featurelist is None:
            featurelist = Defaults.FEATURELIST
        self._image = image_obj
        # self._cond_dict = image_obj._meta_data.well_conditions(self._well_id)
        self._overlay = self._overlay_mask()
        self.image_df = self._combine_channels(featurelist)
        self.quality_df = self._concat_quality_df()

    def _overlay_mask(self) -> pd.DataFrame:
        """Links nuclear IDs with cell IDs"""

        def get_overlap_and_stack(nuclei_mask, other_mask) -> np.ndarray:
            overlap = (nuclei_mask != 0) & (other_mask != 0)
            return np.stack([nuclei_mask[overlap], other_mask[overlap]], axis=1)

        # self._all_masks = {
        #     'CD8_ID': self._image.cd8_mask,
        #     'CD66B_ID': self._image.cd66b_mask,
        #     'CD68_ID': self._image.cd68_mask,
        #     'CD16_ID':self._image.cd16_mask
        # }

        merged_df = None

        for col_name, mask in self._image.masks.items():
            current_df = pd.DataFrame(get_overlap_and_stack(self._image.masks['DAPI'], mask), columns=['label', col_name]).drop_duplicates()
            if merged_df is None:
                merged_df = current_df
            else:
                merged_df = merged_df.merge(current_df, on="label", how="inner")

        return merged_df

    @staticmethod
    def _edit_properties(channel, segment, featurelist):
        """generates a dictionary with """
        feature_dict = {feature: f"{feature}_{channel}_{segment}" for feature in featurelist[2:]}
        feature_dict['area'] = f'area_{segment}'  # the area is the same for each channel
        return feature_dict

    def _get_properties(self, segmentation_mask, channel, segment, featurelist):
        """Measure selected features for each segmented cell in given channel"""
        props = measure.regionprops_table(segmentation_mask, self._image.img_dict[channel], properties=featurelist)
        data = pd.DataFrame(props)
        feature_dict = self._edit_properties(channel, segment, featurelist)
        return data.rename(columns=feature_dict)

    def _channel_data(self, channel, featurelist):
        nucleus_data = self._get_properties(self._image.masks['DAPI'], channel, 'nucleus', featurelist)
        # merge channel data, outer merge combines all area columns into 1
        nucleus_data = pd.merge(nucleus_data, self._overlay, how="outer", on=["label"]).dropna(axis=0, how='any')
        if channel == 'DAPI':
            nucleus_data['integrated_int_DAPI'] = nucleus_data['intensity_mean_DAPI_nucleus'] * nucleus_data[
                'area_nucleus']
        cell_data = self._get_properties(self._image.masks['CD16'], channel, 'CD_16_cell', featurelist)
        # cell_data=pd.merge(cell_data, self._image.data_inter_M, how="outer", on=["label"]).dropna(axis=0, how='any')
        # cyto_data = self._get_properties(self._image.cyto_mask, channel, 'cyto', featurelist)
        merge_1 = pd.merge(nucleus_data,cell_data, how="outer", on=["label"]).dropna(axis=0, how='any')
        # merge_1 = merge_1.rename(columns={'label': 'Cyto_ID'})
        # return pd.merge(nucleus_data, merge_1, how="outer", on=["Cyto_ID"]).dropna(axis=0, how='any')
        return merge_1

    def _combine_channels(self, featurelist):
        channel_data = [self._channel_data(channel, featurelist) for channel in self._image.img_dict.keys()]
        props_data = pd.concat(channel_data, axis=1, join="inner")
        edited_props_data = props_data.loc[:, ~props_data.columns.duplicated()].copy()

        return edited_props_data

    def _set_quality_df(self, channel, corr_img):
        """generates df for image quality control saving the median intensity of the image"""
        return pd.DataFrame({
                             "channel": [channel],
                             "intensity_median": [np.median(corr_img)]})

    def _concat_quality_df(self) -> pd.DataFrame:
        """Concatenate quality dfs for all channels in _corr_img_dict"""
        df_list = [self._set_quality_df(channel, image) for channel, image in self._image.img_dict.items()]
        return pd.concat(df_list)


