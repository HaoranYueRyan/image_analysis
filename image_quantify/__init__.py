
__version__ = '0.1.1'


import pathlib

class Defaults:
    """ store the defaults variables """
    CHANNEL_NAME={0:'DAPI',1:'p-4e-BP1',2:'CD8',3:'CD66b',4:'CD68',5:'CD16',6:'pck',7:'AUTO'}
    DEFAULT_DEST_DIR = "Desktop"  # Decides where the final data folder will be made
    MASK = "Segmentation"
    DATA = "single_img_data"
    QUALITY_CONTROL = "quality_control"
    IMGS_CORR = "images_corrected"
    PLOT_FIGURES = "figures"
    PATH = pathlib.Path.cwd().parent

    MODEL_DICT = {
        'DAPI': 'nuclei',
        'p-4e-BP1': 'None',
        'CD8': 'cyto',
        'CD66b': 'cyto',
        'CD68': 'cyto',
        'CD16': 'cyto',
        'pck': 'SAM',
        'AUTO': 'None',
    }
    FEATURELIST = ['label', 'area', 'intensity_max', 'intensity_mean']

SEPARATOR = "==========================================================================================\n"



