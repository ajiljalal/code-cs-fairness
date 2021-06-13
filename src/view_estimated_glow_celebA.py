"""View estimated images for celebA"""
# pylint: disable = C0301, R0903, R0902

import numpy as np
import celebA_input
from celebA_utils import view_image
import utils
import matplotlib.pyplot as plt
import pickle as pkl


class Hparams(object):
    """Hyperparameters"""
    def __init__(self):
        self.input_type = 'full-input'
        self.input_path_pattern = './test_images/celebA/*.jpg'
        self.input_path = './test_images/celebA'
        self.num_input_images = 30
        self.image_matrix = 0
        self.image_shape = (3,256,256)
        self.image_size = 256
        self.n_input = np.prod(self.image_shape)
        self.measurement_type = 'circulant'
        self.model_types = ['MAP','Langevin']


def view(xs_dict, patterns_images, patterns_lpips, patterns_l2, images_nums, hparams, **kws):
    """View the images"""
    x_hats_dict = {}
    lpips_dict = {}
    l2_dict = {}
    for model_type, pattern_image, pattern_lpips, pattern_l2 in zip(hparams.model_types, patterns_images, patterns_lpips, patterns_l2):
        outfiles = [pattern_image.format(i) for i in images_nums]
        x_hats_dict[model_type] = {i: plt.imread(outfile) for i, outfile in enumerate(outfiles)}
        with open(pattern_lpips, 'rb') as f:
            lpips_dict[model_type] = pkl.load(f)
        with open(pattern_l2, 'rb') as f:
            l2_dict[model_type] = pkl.load(f)
    xs_dict_temp = {i : xs_dict[i] for i in images_nums}
    utils.image_matrix(xs_dict_temp, x_hats_dict, lpips_dict, l2_dict, view_image, hparams, **kws)


def get_image_nums(start, stop, hparams):
    """Get range of images"""
    assert start >= 0
    assert stop <= hparams.num_input_images
    images_nums = list(range(start, stop))
    return images_nums


def main():
    """Make and save image matrices"""
    hparams = Hparams()
    xs_dict = celebA_input.model_input(hparams)
    start, stop = 0, 5
    images_nums = get_image_nums(start, stop, hparams)
    is_save = True
    for num_measurements in [3072]:# [2500, 5000, 10000, 15000, 20000, 25000, 30000, 35000]:
        for proper in [True, False]:
            # if proper:
            #     pattern1_base = './estimated/celebA/full-input/circulant/16.0/' + str(num_measurements) + '/glow_map/None_None_adam_0.001_0.0_3000_1/'
            # else:
            #     pattern1_base = './estimated/celebA/full-input/circulant/16.0/' + str(num_measurements) + '/glow_map/None_0.001_adam_0.001_0.0_3000_1/'

            pattern1_base = 'estimated/blonde/full-input/superres/2.0/3072/glow_map/1.0_0.002657312925170068_adam_0.001_0.0_3000_1/'
            pattern1_images = pattern1_base + '{0}.png'
            pattern1_lpips = pattern1_base + 'lpips_scores.pkl'
            pattern1_l2 = pattern1_base + 'l2_losses.pkl'

            # if proper:
            #     pattern2_base = './estimated/celebA/full-input/circulant/16.0/' + str(num_measurements) + '/glow_langevin/None_None_None_sgd_1e-05_0.0_3000_1/'
            # else:
            #     pattern2_base = './estimated/celebA/full-input/circulant/16.0/' + str(num_measurements) + '/glow_langevin/None_0.1_None_sgd_1e-05_0.0_3000_1/'

            pattern2_base = 'estimated/blonde/full-input/superres/2.0/3072/glow_langevin/384.0_1.0204081632653061_0.00447213595499958_sgd_1e-05_0.0_3000_1/'
            pattern2_images = pattern2_base + '{0}.png'
            pattern2_lpips = pattern2_base + 'lpips_scores.pkl'
            pattern2_l2 = pattern2_base + 'l2_losses.pkl'
            patterns_images = [pattern1_images, pattern2_images]
            patterns_lpips = [pattern1_lpips, pattern2_lpips]
            patterns_l2 = [pattern1_l2, pattern2_l2]
            try:
                view(xs_dict, patterns_images, patterns_lpips, patterns_l2, images_nums, hparams)
            except:
                continue
            if proper:
                # base_path = './results/celebA_reconstr_{}_orig_glow_map_langevin_proper.pdf'
                base_path = './results/blonde_reconstr_{}_orig_glow_map_langevin_proper.pdf'
            else:
                # base_path = './results/celebA_reconstr_{}_orig_glow_map_langevin_improper.pdf'
                base_path = './results/blonde_reconstr_{}_orig_glow_map_langevin_improper.pdf'

            save_path = base_path.format(num_measurements)
            utils.save_plot(is_save, save_path)

if __name__ == '__main__':
    main()
