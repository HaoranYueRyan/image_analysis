




import pandas as pd
import pathlib
from image_quantify import Defaults, SEPARATOR
class ExpPaths:
    def __init__(self):
        self._create_dir_paths()
        self._create_exp_dir()

    def _create_dir_paths(self):
        """ Generate path attributes for experiment"""
        self.path = pathlib.Path.home() / Defaults.DEFAULT_DEST_DIR / f"{'Quantify_intensity_level'}"
        self.Mask = self.path / Defaults.MASK
        self.final_data = self.path / Defaults.DATA
        self.quality_ctr = self.path / Defaults.QUALITY_CONTROL
        self.example_img = self.path / Defaults.IMGS_CORR
        self.figures = self.path / Defaults.PLOT_FIGURES

    def _create_exp_dir(self):
        path_list = [self.path,  self.final_data,self.Mask,
                     self.quality_ctr, self.example_img, self.figures]
        for path in path_list:
            path.mkdir(exist_ok=True)

        print(f'Gathering data and assembling directories for experiment \n{SEPARATOR}')


if __name__=='__main__':
    exp=ExpPaths()
