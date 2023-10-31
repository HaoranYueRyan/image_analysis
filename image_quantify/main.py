import skimage
from image_quantify import Defaults
import numpy as np
from image_quantify.image_analysis import Image,ImageProperties
from image_quantify.data_structure import ExpPaths


def main(image_path):
    exp_paths=ExpPaths()
    # stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    ori_img=skimage.io.imread(image_path)
    image_dict={}
    # for count, well in enumerate(list(meta_data.plate_obj.listChildren())):
    for channel_num in range(ori_img.shape[0]):
        image_dict[Defaults.CHANNEL_NAME[int(channel_num)]]=np.array(ori_img[channel_num,:,:])
    img=Image(image_dict,exp_paths)
    img.segmentation_figure()
    image_data=ImageProperties(img)
    df_final=image_data.image_df
    df_final.to_csv(exp_paths.final_data / "analysis_final_data.csv")
    print('finished quantify the data')


if __name__ == '__main__':
    main('/Users/haoranyue/Documents/P269_HumTORMP_TMA14_Scan1_component_data.tif')




