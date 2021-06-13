"""View estimated images for ffhq-69000"""
# pylint: disable = C0301, R0903, R0902

import numpy as np
import celebA_input
from celebA_utils import view_image
import utils
import matplotlib.pyplot as plt
import pickle as pkl
from metrics_utils import int_or_float, find_best
import glob

class Hparams(object):
    """Hyperparameters"""
    def __init__(self):
        self.input_type = 'full-input'
        #self.input_path_pattern = './test_images/celebA'
        #self.input_path = './test_images/celebA'
        self.input_path_pattern = './test_images/ffhq-69000/*.jpg'
        self.input_path = './test_images/ffhq-69000'
        self.num_input_images = 30
        self.image_matrix = 0
        self.image_shape = (3,256,256)
        self.image_size = 256
        self.noise_std = 0.0
        #self.noise_std = 16.0
        self.n_input = np.prod(self.image_shape)
        self.measurement_type = 'circulant'
        self.model_types = ['MAP', 'Deep-Decoder','Langevin']
        #self.model_types = ['MAP', 'Modified-MAP', 'Langevin']


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
# def view(xs_dict, patterns, images_nums, hparams, **kws):
#     """View the images"""
#     x_hats_dict = {}
#     for model_type, pattern in zip(hparams.model_types, patterns):
#         outfiles = [pattern.format(i) for i in images_nums]
#         x_hats_dict[model_type] = {i: plt.imread(outfile) for i, outfile in enumerate(outfiles)}
#     xs_dict_temp = {i : xs_dict[i] for i in images_nums}
#     utils.image_matrix(xs_dict_temp, x_hats_dict, view_image, hparams, **kws)


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

    def formatted(f):
        return format(f, '.4f').rstrip('0').rstrip('.')
    legend_base_regexs = [
        ('MAP',
                f'./estimated/ffhq-69000/full-input/circulant/{hparams.noise_std}/',
                     '/ncsnv2/map/*'),
        ('Deep-Decoder',
                f'./estimated/ffhq-69000/full-input/circulant/{hparams.noise_std}/',
                     '/dd/map/*'),
            ('Langevin',
                    f'./estimated/ffhq-69000/full-input/circulant/{hparams.noise_std}/',
                         '/ncsnv2/langevin/*')

    ]
    #legend_base_regexs = [
    #    ('MAP',
    #f'./estimated*/celebA/full-input/circulant/{hparams.noise_std}/',
    # '/glow*map*/*'),
    #    ('Modified-MAP',
    #f'./estimated*/celebA/full-input/circulant/{hparams.noise_std}/',
    # '/glow*map*/*'),
    #    ('Langevin',
    #f'./estimated*/celebA/full-input/circulant/{hparams.noise_std}/',
    # '/glow*langevin/None_None*'),
    #             ]
    retrieve_list = [['lpips', 'mean'], ['lpips', 'std']]

    #for num_measurements in [2500,5000,10000,15000,20000,30000,35000] :
    for num_measurements in [5000,10000,15000,40000,50000,75000]:
    #for num_measurements in [100,200,500,1000,2500,5000,7500,10000]:
        patterns_images, patterns_images_2, patterns_lpips, patterns_l2 = [], [] , [], []
        exists = True
        for legend, base, regex in legend_base_regexs:
            keys = map(int_or_float, [a.split('/')[-1] for a in glob.glob(base + '*')])
            list_keys = [key for key in keys]
            print(list_keys)
            if num_measurements not in list_keys:
                exists = False
                break
            pattern = base + str(num_measurements) + regex
            if 'glow' in regex and legend in [ 'MAP', 'Langevin']:
                criterion = ['likelihood', 'mean']
            else:
                criterion = ['l2', 'mean']

            _, best_dir = find_best(pattern, criterion, retrieve_list)
            print(best_dir)
            pattern_images = best_dir + '/{0}.png'
            pattern_images_2 = best_dir + '/images/{:06d}.png'
            pattern_lpips = best_dir + '/lpips_scores.pkl'
            pattern_l2 = best_dir + '/l2_losses.pkl'
            patterns_images.append(pattern_images)
            patterns_images_2.append(pattern_images_2)
            patterns_lpips.append(pattern_lpips)
            patterns_l2.append(pattern_l2)
        print(patterns_images)
        if exists:
            try:
                view(xs_dict, patterns_images, patterns_lpips, patterns_l2, images_nums, hparams)
            except FileNotFoundError:
                view(xs_dict, patterns_images_2, patterns_lpips, patterns_l2, images_nums, hparams)
            except FileNotFoundError:
                pass

            # patterns = [pattern2, pattern3]
            # view(xs_dict, patterns, images_nums, hparams)
            save_path = f'./results/ffhq-69000_reconstr_{num_measurements}_{criterion[0]}_ncsnv2_orig_map_langevin.pdf'
            #save_path = f'./results/celebA_reconstr_{num_measurements}_{criterion[0]}_glow_orig_map_langevin.pdf'
            utils.save_plot(is_save, save_path)
        else:
            continue
        # except:
        #     pass

if __name__ == '__main__':
    main()
