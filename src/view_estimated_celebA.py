"""View estimated images for ffhq-69000"""
# pylint: disable = C0301, R0903, R0902

import numpy as np
import utils
import matplotlib.pyplot as plt
import pickle as pkl
from metrics_utils import int_or_float, find_best
import glob

class Hparams(object):
    """Hyperparameters"""
    def __init__(self):
        self.input_type = 'full-input'
        self.input_path_pattern = './test_images/celebA'
        self.input_path = './test_images/celebA'
        self.num_input_images = 30
        self.image_matrix = 0
        self.image_shape = (3,256,256)
        self.image_size = 256
        self.noise_std = 16.0
        self.n_input = np.prod(self.image_shape)
        self.measurement_type = 'circulant'
        self.model_types = ['MAP', 'Langevin']


def view(xs_dict, patterns_images, patterns_l2, images_nums, hparams, **kws):
    """View the images"""
    x_hats_dict = {}
    l2_dict = {}
    for model_type, pattern_image, pattern_l2 in zip(hparams.model_types, patterns_images, patterns_l2):
        outfiles = [pattern_image.format(i) for i in images_nums]
        x_hats_dict[model_type] = {i: plt.imread(outfile) for i, outfile in enumerate(outfiles)}
        with open(pattern_l2, 'rb') as f:
            l2_dict[model_type] = pkl.load(f)
    xs_dict_temp = {i : xs_dict[i] for i in images_nums}
    utils.image_matrix(xs_dict_temp, x_hats_dict, l2_dict, hparams, **kws)


def get_image_nums(start, stop, hparams):
    """Get range of images"""
    assert start >= 0
    assert stop <= hparams.num_input_images
    images_nums = list(range(start, stop))
    return images_nums


def main():
    """Make and save image matrices"""
    hparams = Hparams()
    xs_dict = utils.model_input(hparams)
    start, stop = 0, 1
    images_nums = get_image_nums(start, stop, hparams)
    is_save = True

    def formatted(f):
        return format(f, '.4f').rstrip('0').rstrip('.')
    legend_base_regexs = [
        ('MAP',
    f'./estimated/celebA/full-input/circulant/{hparams.noise_std}/',
     '/glow/map/None_None*'),
        ('Langevin',
    f'./estimated*/celebA/full-input/circulant/{hparams.noise_std}/',
     '/glow/*langevin/None*'),
                 ]
    retrieve_list = [['l2', 'mean'], ['l2', 'std']]

    for num_measurements in [2500,5000,10000,20000,30000,35000] :
        patterns_images, patterns_l2 = [], []
        exists = True
        for legend, base, regex in legend_base_regexs:
            keys = map(int_or_float, [a.split('/')[-1] for a in glob.glob(base + '*')])
            list_keys = [key for key in keys]
            if num_measurements not in list_keys:
                exists = False
                break
            pattern = base + str(num_measurements) + regex
            if 'glow' in regex and legend in [ 'MAP', 'Langevin']:
                criterion = ['likelihood', 'mean']
            else:
                criterion = ['l2', 'mean']

            _, best_dir = find_best(pattern, criterion, retrieve_list)
            pattern_images = best_dir + '/images/{:06d}.png'
            pattern_l2 = best_dir + '/l2_losses.pkl'
            patterns_images.append(pattern_images)
            patterns_l2.append(pattern_l2)
        if exists:
            view(xs_dict, patterns_images, patterns_l2, images_nums, hparams)
            save_path = f'./results/celebA-reconstr-{num_measurements}-{criterion[0]}.pdf'
            utils.save_plot(is_save, save_path)
        else:
            print(f'Could not find reconstructions for {num_measurements}')

if __name__ == '__main__':
    main()
