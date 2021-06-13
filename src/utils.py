"""Some common utils"""
# pylint: disable = C0301, C0103, C0111

import os
import pickle
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.transform import downscale_local_mean
import sys

try:
	import tensorflow as tf
except ImportError:
	pass
import celebA_estimators
font = { 'weight' : 'bold',
        'size'   : 30}
import matplotlib
matplotlib.rc('font', **font)

#sys.path.append(os.path.join(os.path.dirname(__file__), 'bm3d_demos'))
#from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr

class BestKeeper(object):
    """Class to keep the best stuff"""
    def __init__(self, batch_size, n_input):
        self.batch_size = batch_size
        self.losses_val_best = [1e100 for _ in range(batch_size)]
        self.x_hat_batch_val_best = np.zeros((batch_size, n_input))

    def report(self, x_hat_batch_val, losses_val):
        for i in range(self.batch_size):
            if losses_val[i] < self.losses_val_best[i]:
                self.x_hat_batch_val_best[i, :] = x_hat_batch_val[i, :]
                self.losses_val_best[i] = losses_val[i]

    def get_best(self):
        return self.x_hat_batch_val_best


def get_l2_loss(image1, image2):
    """Get L2 loss between the two images"""
    assert image1.shape == image2.shape
    return np.mean((image1 - image2)**2)


def get_measurement_loss(x_hat, A, y, hparams):
    """Get measurement loss of the estimated image"""
    y_hat = get_measurements(x_hat, A, 0 , hparams)
    # measurements are in a batch of size 1.
    y_hat = y_hat.reshape(y.shape)
    # if A is None:
    #     y_hat = x_hat
    # else:
    #     y_hat = np.matmul(x_hat, A)
    assert y_hat.shape == y.shape
    return np.sum((y - y_hat) ** 2)

def get_lpips_score(percept, x_hat, x, image_shape):
    device = 'cuda:0'
    x_hat_tensor = torch.Tensor(x_hat.reshape((1,) + image_shape)).to(device)
    x_tensor = torch.Tensor(x.reshape((1,) + image_shape)).to(device)

    score = percept(x_hat_tensor, x_tensor).sum().item()
    return score

def save_to_pickle(data, pkl_filepath):
    """Save the data to a pickle file"""
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_if_pickled(pkl_filepath):
    """Load if the pickle file exists. Else return empty dict"""
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data


def get_estimator(hparams, model_type):
    if model_type == 'map' and hparams.net == 'realnvp':
        estimator = celebA_estimators.realnvp_map_estimator(hparams)
    elif model_type == 'xmap' and hparams.net == 'realnvp':
        estimator = celebA_estimators.realnvp_xmap_estimator(hparams)
    elif model_type == 'noisy' and hparams.net == 'realnvp':
        estimator = celebA_estimators.noisy_estimator(hparams)
    elif model_type == 'langevin' and hparams.net == 'realnvp':
        estimator = celebA_estimators.realnvp_langevin_estimator(hparams)
    elif model_type == 'xlangevin' and hparams.net == 'realnvp':
        estimator = celebA_estimators.xlangevin_estimator(hparams)
    elif model_type == 'map' and hparams.net == 'glow':
        estimator = celebA_estimators.glow_annealed_map_estimator(hparams)
    elif model_type == 'langevin' and hparams.net == 'glow':
        estimator = celebA_estimators.glow_annealed_langevin_estimator(hparams)
    elif model_type == 'langevin' and hparams.net == 'stylegan2':
        estimator = celebA_estimators.stylegan_langevin_estimator(hparams)
    elif model_type == 'pulse':
        assert hparams.net.lower() == 'stylegan2'
        estimator = celebA_estimators.stylegan_pulse_estimator(hparams)
    elif model_type == 'map' and hparams.net == 'stylegan2':
        estimator = celebA_estimators.stylegan_map_estimator(hparams)
    elif model_type == 'map' and hparams.net == 'ncsnv2':
        estimator = celebA_estimators.ncsnv2_langevin_estimator(hparams, MAP=True)
    elif model_type == 'langevin' and hparams.net == 'ncsnv2':
        estimator = celebA_estimators.ncsnv2_langevin_estimator(hparams, MAP=False)
    elif hparams.net == 'dd':
        estimator = celebA_estimators.deep_decoder_estimator(hparams)
    else:
        raise NotImplementedError
    return estimator


def get_estimators(hparams):
    estimators = {model_type: get_estimator(hparams, model_type) for model_type in hparams.model_types}
    return estimators


def setup_checkpointing(hparams):
    # Set up checkpoint directories
    for model_type in hparams.model_types:
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        set_up_dir(checkpoint_dir)


def save_images(est_images, save_image, hparams):
    """Save a batch of images to png files"""
    for model_type in hparams.model_types:
        for image_num, image in est_images[model_type].items():
            save_path = get_save_paths(hparams, image_num)[model_type]
            image = image.reshape(hparams.image_shape)
            save_image(image, save_path)


def checkpoint(est_images, measurement_losses, l2_losses, lpips_scores, z_hats, likelihoods, save_image, hparams):
    """Save images, measurement losses and L2 losses for a batch"""
    if hparams.save_images:
        save_images(est_images, save_image, hparams)

    if hparams.save_stats:
        for model_type in hparams.model_types:
            m_losses_filepath, l2_losses_filepath, lpips_scores_filepath, z_hats_filepath, likelihoods_filepath = get_pkl_filepaths(hparams, model_type)
            save_to_pickle(measurement_losses[model_type], m_losses_filepath)
            save_to_pickle(l2_losses[model_type], l2_losses_filepath)
            save_to_pickle(lpips_scores[model_type], lpips_scores_filepath)
            save_to_pickle(z_hats[model_type], z_hats_filepath)
            save_to_pickle(likelihoods[model_type], likelihoods_filepath)


def load_checkpoints(hparams):
    measurement_losses, l2_losses, lpips_scores, z_hats, likelihoods = {}, {}, {}, {}, {}
    if hparams.save_images:
        # Load pickled loss dictionaries
        for model_type in hparams.model_types:
            m_losses_filepath, l2_losses_filepath, lpips_scores_filepath, z_hats_filepath, likelihoods_filepath  = get_pkl_filepaths(hparams, model_type)
            measurement_losses[model_type] = load_if_pickled(m_losses_filepath)
            l2_losses[model_type] = load_if_pickled(l2_losses_filepath)
            lpips_scores[model_type] = load_if_pickled(lpips_scores_filepath)
            z_hats[model_type] = load_if_pickled(z_hats_filepath)
            likelihoods[model_type] = load_if_pickled(likelihoods_filepath)
    else:
        for model_type in hparams.model_types:
            measurement_losses[model_type] = {}
            l2_losses[model_type] = {}
            lpips_scores[model_type] = {}
            z_hats[model_type] = {}
            likelihoods[model_type] = {}
    return measurement_losses, l2_losses, lpips_scores, z_hats, likelihoods


def image_matrix(images, est_images, lpips_scores, l2_losses, view_image, hparams, alg_labels=True):
    """Display images"""

    if hparams.measurement_type in ['inpaint', 'superres']:
        figure_height = 2 + len(hparams.model_types)
    else:
        figure_height = 1 + len(hparams.model_types)

    fig = plt.figure(figsize=[4*len(images), 4.3*figure_height])

    outer_counter = 0
    inner_counter = 0

    # Show original images
    outer_counter += 1
    for image in images.values():
        inner_counter += 1
        ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])
        if alg_labels:
            ax.set_ylabel('Original')#, fontsize=14)
        _ = fig.add_subplot(figure_height, len(images), inner_counter)
        view_image(image, hparams)

    # Show original images with inpainting mask
    if hparams.measurement_type == 'inpaint':
        mask = get_inpaint_mask(hparams)
        outer_counter += 1
        for image in images.values():
            inner_counter += 1
            ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            if alg_labels:
                ax.set_ylabel('Masked') #, fontsize=14)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image, hparams, mask)

    # Show original images with blurring
    if hparams.measurement_type == 'superres':
        factor = hparams.superres_factor
        A = get_A_superres(hparams)
        outer_counter += 1
        for image in images.values():
            image_low_res = np.matmul(image, A) / np.sqrt(hparams.n_input/(factor**2)) / (factor**2)
            low_res_shape = (int(hparams.image_shape[0]/factor), int(hparams.image_shape[1]/factor), hparams.image_shape[2])
            image_low_res = np.reshape(image_low_res, low_res_shape)
            inner_counter += 1
            ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            if alg_labels:
                ax.set_ylabel('Blurred') #, fontsize=14)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            view_image(image_low_res, hparams)

    for model_type in hparams.model_types:
        outer_counter += 1
        for image, lpips, l2 in zip(est_images[model_type].values(), lpips_scores[model_type].values(), l2_losses[model_type].values()):
            inner_counter += 1
            ax = fig.add_subplot(figure_height, 1, outer_counter, frameon=False)
            # ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            ax.get_xaxis().set_ticks([])
            if alg_labels:
                ax.set_ylabel(model_type) #, fontsize=14)
            _ = fig.add_subplot(figure_height, len(images), inner_counter)
            #_.set_title(f'PSNR={-10*np.log10(l2):.3f}\nLPIPS:{lpips:.3f}', fontsize=8)
            view_image(image, hparams)

    if hparams.image_matrix >= 2:
        save_path = get_matrix_save_path(hparams)
        plt.savefig(save_path)

    if hparams.image_matrix in [1, 3]:
        plt.show()


def plot_image(image, cmap=None):
    """Show the image"""
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame = frame.imshow(image, cmap=cmap)


def get_checkpoint_dir(hparams, model_type):
    if hparams.annealed :
        base_dir = './estimated/{0}/{1}/{2}/{3}/{4}/{5}/annealed_{6}/'.format(
            hparams.dataset,
            hparams.input_type,
            hparams.measurement_type,
            hparams.noise_std,
            hparams.num_measurements,
            hparams.net,
            model_type
        )
    else:
        base_dir = './estimated/{0}/{1}/{2}/{3}/{4}/{5}/{6}/'.format(
            hparams.dataset,
            hparams.input_type,
            hparams.measurement_type,
            hparams.noise_std,
            hparams.num_measurements,
            hparams.net,
            model_type
        )


    if hparams.net == 'ncsnv2':
            dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                hparams.mloss_weight,
                hparams.ncsnv2_configs['sampling']['step_lr'],
                hparams.ncsnv2_configs['sampling']['n_steps_each'],
                hparams.ncsnv2_configs['model']['sigma_begin'],
                int(hparams.L),
                hparams.ncsnv2_configs['model']['ema'],
                hparams.ncsnv2_configs['model']['ema_rate'],
                hparams.ncsnv2_configs['model']['sigma_dist'],
                hparams.ncsnv2_configs['model']['sigma_end']
            )
    elif hparams.net == 'dd':
            dir_name = '{}_{}_{}'.format(
                hparams.optimizer_type,
                hparams.learning_rate,
                hparams.max_update_iter
            )

    else:

        if model_type in ['map'] and (not hparams.annealed):
            dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                hparams.mloss_weight,
                hparams.zprior_weight,
                hparams.fixed_init,
                hparams.optimizer_type,
                hparams.project,
                hparams.learning_rate,
                hparams.momentum,
                hparams.max_update_iter,
                hparams.num_random_restarts,
            )
        elif model_type in ['map'] and hparams.annealed:
            if hparams.net == 'stylegan2':
                dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                    hparams.zprior_weight,
                    hparams.T,
                    hparams.L,
                    hparams.sigma_init,
                    hparams.sigma_final,
                    hparams.fixed_init,
                    hparams.optimizer_type,
                    hparams.project,
                    hparams.learning_rate,
                    hparams.momentum,
                    hparams.max_update_iter,
                    hparams.num_random_restarts,
                    hparams.lpips,
                    hparams.num_noise_variables
                )
            else:
                dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                    hparams.zprior_weight,
                    hparams.T,
                    hparams.L,
                    hparams.sigma_init,
                    hparams.sigma_final,
                    hparams.fixed_init,
                    hparams.optimizer_type,
                    hparams.learning_rate,
                    hparams.momentum,
                    hparams.max_update_iter,
                    hparams.num_random_restarts,
                )
        elif model_type in ['xmap'] and hparams.annealed:
            dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                hparams.ploss_weight,
                hparams.T,
                hparams.L,
                hparams.sigma_init,
                hparams.sigma_final,
                hparams.fixed_init,
                hparams.optimizer_type,
                hparams.learning_rate,
                hparams.momentum,
                hparams.max_update_iter,
                hparams.num_random_restarts,
            )
        elif model_type in ['langevin'] and not hparams.annealed:
            dir_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
                hparams.mloss_weight,
                hparams.zprior_weight,
                hparams.gradient_noise_weight,
                hparams.optimizer_type,
                hparams.learning_rate,
                hparams.momentum,
                hparams.max_update_iter,
                hparams.num_random_restarts,
            )
        elif model_type in ['noisy']:
            dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                hparams.mloss_weight,
                hparams.ploss_weight,
                hparams.zprior_weight,
                hparams.gradient_noise_weight,
                hparams.optimizer_type,
                hparams.learning_rate,
                hparams.momentum,
                hparams.max_update_iter,
                hparams.num_random_restarts,
            )
        elif model_type in ['langevin'] and hparams.annealed:
            if hparams.net == 'stylegan2':
                dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                    hparams.zprior_weight,
                    hparams.gradient_noise_weight,
                    hparams.T,
                    hparams.L,
                    hparams.sigma_init,
                    hparams.sigma_final,
                    hparams.fixed_init,
                    hparams.zprior_init_sdev,
                    hparams.zprior_sdev,
                    hparams.optimizer_type,
                    hparams.learning_rate,
                    hparams.momentum,
                    hparams.max_update_iter,
                    hparams.num_random_restarts,
                    hparams.lpips,
                    hparams.num_noise_variables
                )
            else:
                dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                    hparams.zprior_weight,
                    hparams.gradient_noise_weight,
                    hparams.T,
                    hparams.L,
                    hparams.sigma_init,
                    hparams.sigma_final,
                    hparams.fixed_init,
                    hparams.zprior_init_sdev,
                    hparams.zprior_sdev,
                    hparams.optimizer_type,
                    hparams.learning_rate,
                    hparams.momentum,
                    hparams.max_update_iter,
                    hparams.num_random_restarts
                )
        elif model_type in ['xlangevin'] and hparams.annealed:
            dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                hparams.ploss_weight,
                hparams.gradient_noise_weight,
                hparams.T,
                hparams.L,
                hparams.sigma_init,
                hparams.sigma_final,
                hparams.fixed_init,
                hparams.optimizer_type,
                hparams.learning_rate,
                hparams.momentum,
                hparams.max_update_iter,
                hparams.num_random_restarts,
            )


        elif model_type in ['pulse']:
            dir_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                    hparams.seed,
                    hparams.loss_str,
                    hparams.pulse_eps,
                    hparams.noise_type,
                    hparams.tile_latent,
                    hparams.num_noise_variables,
                    hparams.optimizer_type,
                    hparams.learning_rate,
                    hparams.max_update_iter,
                    hparams.lr_schedule,
                    )
        else:
            raise NotImplementedError

    ckpt_dir = base_dir + dir_name + '/'

    return ckpt_dir


def get_pkl_filepaths(hparams, model_type):
    """Return paths for the pickle files"""
    checkpoint_dir = get_checkpoint_dir(hparams, model_type)
    m_losses_filepath = checkpoint_dir + 'measurement_losses.pkl'
    l2_losses_filepath = checkpoint_dir + 'l2_losses.pkl'
    lpips_scores_filepath = checkpoint_dir + 'lpips_scores.pkl'
    z_hats_filepath = checkpoint_dir + 'z.pkl'
    likelihoods_filepath = checkpoint_dir + 'likelihoods.pkl'
    return m_losses_filepath, l2_losses_filepath, lpips_scores_filepath, z_hats_filepath, likelihoods_filepath


def get_save_paths(hparams, image_num):
    save_paths = {}
    for model_type in hparams.model_types:
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        image_dir = os.path.join(checkpoint_dir, 'images')
        set_up_dir(image_dir)
        save_paths[model_type] = os.path.join(image_dir , '{0:06d}.png'.format(image_num))
    return save_paths


def get_matrix_save_path(hparams):
    save_path = './estimated/{0}/{1}/{2}/{3}/{4}/matrix_{5}.png'.format(
        hparams.dataset,
        hparams.input_type,
        hparams.measurement_type,
        hparams.noise_std,
        hparams.num_measurements,
        '_'.join(hparams.model_types)
    )
    return save_path


def set_up_dir(directory, clean=False):
    if os.path.exists(directory):
        if clean:
            shutil.rmtree(directory)
    else:
        os.makedirs(directory)


def print_hparams(hparams):
    print('')
    for temp in dir(hparams):
        if temp[:1] != '_':
            print('{0} = {1}'.format(temp, getattr(hparams, temp)))
    print('')


def get_learning_rate(global_step, hparams):
    if hparams.decay_lr:
        return tf.train.exponential_decay(hparams.learning_rate,
                                          global_step,
                                          50,
                                          0.7,
                                          staircase=True)
    else:
        return tf.constant(hparams.learning_rate)


def get_optimizer(z, learning_rate, hparams):
    if hparams.optimizer_type == 'sgd':
        if hparams.net == 'realnvp':
            return torch.optim.SGD([z], learning_rate, momentum=hparams.momentum)
        elif hparams.net == 'stylegan2':
            return torch.optim.SGD(z, learning_rate, momentum=hparams.momentum)

        elif hparams.net == 'glow':
            if hparams.momentum == 0.:
                return tf.train.GradientDescentOptimizer(learning_rate)
            else:
                return tf.train.MomentumOptimizer(learning_rate, hparams.momentum)
    elif hparams.optimizer_type == 'rmsprop':
        return torch.optim.RMSprop([z], lr=learning_rate, momentum=hparams.momentum)
    elif hparams.optimizer_type == 'adam':
        if hparams.net == 'realnvp':
            return torch.optim.Adam([z], lr=learning_rate)
        elif hparams.net == 'stylegan2':
            return torch.optim.Adam(z, lr=learning_rate)
        elif hparams.net == 'glow':
            return tf.train.AdamOptimizer(learning_rate)
    elif hparams.optimizer_type == 'adagrad':
        return torch.optim.Adagrad([z], lr=learning_rate)
    elif hparams.optimizer_type == 'lbfgs':
        if hparams.net == 'realnvp':
            return torch.optim.LBFGS([z], lr=learning_rate)
        elif hparams.net == 'glow':
            return Exception('Tensorflow does not support ' + hparams.optimizer_type)
    else:
        raise Exception('Optimizer ' + hparams.optimizer_type + ' not supported')


def get_inpaint_mask(hparams):
    image_size = hparams.image_shape[0]
    margin = (image_size - hparams.inpaint_size) / 2
    mask = np.ones(hparams.image_shape)
    mask[margin:margin+hparams.inpaint_size, margin:margin+hparams.inpaint_size] = 0
    return mask


def get_A_inpaint(hparams):
    mask = get_inpaint_mask(hparams)
    mask = mask.reshape(1, -1)
    A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
    A = np.asarray([a for a in A if np.sum(a) != 0])

    # Make sure that the norm of each row of A is hparams.n_input
    A = np.sqrt(hparams.n_input) * A
    assert all(np.abs(np.sum(A**2, 1) - hparams.n_input) < 1e-6)

    return A.T


def get_A_superres(hparams):
    factor = hparams.superres_factor
    A = np.zeros((int(hparams.n_input/(factor**2)), hparams.n_input))
    l = 0
    for i in range(hparams.image_shape[0]/factor):
        for j in range(hparams.image_shape[1]/factor):
            for k in range(hparams.image_shape[2]):
                a = np.zeros(hparams.image_shape)
                a[factor*i:factor*(i+1), factor*j:factor*(j+1), k] = 1
                A[l, :] = np.reshape(a, [1, -1])
                l += 1

    # Make sure that the norm of each row of A is hparams.n_input
    A = np.sqrt(hparams.n_input/(factor**2)) * A
    assert all(np.abs(np.sum(A**2, 1) - hparams.n_input) < 1e-6)

    return A.T

def get_A(hparams):
    if hparams.measurement_type == 'gaussian':
        A = np.random.randn(hparams.n_input, hparams.num_measurements)/np.sqrt(hparams.num_measurements)
    elif hparams.measurement_type == 'superres':
        A = None
        # A = get_A_superres(hparams)
    elif hparams.measurement_type == 'inpaint':
        A = get_A_inpaint(hparams)
    elif hparams.measurement_type == 'project':
        A = None
    elif hparams.measurement_type == 'circulant':
        temp = np.random.randn(1, hparams.n_input)
        A = temp/ np.sqrt(hparams.num_measurements)
    else:
        raise NotImplementedError
    return A


def set_num_measurements(hparams):
    if hparams.measurement_type == 'project':
        hparams.num_measurements = hparams.n_input
    else:
        hparams.num_measurements = get_A(hparams).shape[1]


def get_checkpoint_path(ckpt_dir):
    ckpt_dir = os.path.abspath(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = os.path.join(ckpt_dir,
                                 ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')
        ckpt_path = ''
    return ckpt_path




def save_plot(is_save, save_path):
    if is_save:
        pdf = PdfPages(save_path)
        pdf.savefig(bbox_inches='tight')
        pdf.close()

def get_opt_reinit_op(opt, var_list, global_step):
    opt_slots = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in var_list]
    if isinstance(opt, tf.train.AdamOptimizer):
        opt_slots.extend([opt._beta1_power, opt._beta2_power])  #pylint: disable = W0212
    all_opt_variables = opt_slots + var_list + [global_step]
    opt_reinit_op = tf.variables_initializer(all_opt_variables)
    return opt_reinit_op


def partial_circulant_tf(inputs, filters, indices, sign_pattern):
    n = np.prod(inputs.get_shape().as_list()[1:])
    bs = inputs.get_shape().as_list()[0]
    input_reshape = tf.reshape(inputs, (-1,n))
    input_sign = tf.multiply(input_reshape , sign_pattern)

    zeros_input = tf.zeros_like(input_sign)
    zeros_filter = tf.zeros_like(filters)
    complex_input = tf.complex(input_sign, zeros_input )
    complex_filter = tf.complex(filters, zeros_filter)
    output_fft = tf.multiply(tf.fft(complex_input), tf.fft(complex_filter))
    output_ifft = tf.ifft(output_fft)
    output = tf.real(output_ifft)
    return tf.gather(output, indices, axis=1)

def partial_circulant_torch(inputs, filters, indices, sign_pattern):
    n = np.prod(inputs.shape[1:])
    bs = inputs.shape[0]
    input_reshape = inputs.view(bs,n)
    input_sign = input_reshape * sign_pattern

    def to_complex(tensor):
        zeros = torch.zeros_like(tensor)
        concat = torch.cat((tensor, zeros), axis=0)
        reshape = concat.view(2,-1,n)
        return reshape.permute(1,2,0)

    complex_input = to_complex(input_sign)
    complex_filter = to_complex(filters)
    input_fft = torch.fft(complex_input, 1)
    filter_fft = torch.fft(complex_filter, 1)
    output_fft = torch.zeros_like(input_fft)
    # is there a simpler way to do complex multiplies in pytorch?
    output_fft[:,:,0] = input_fft[:,:,0]*filter_fft[:,:,0] - input_fft[:,:,1] * filter_fft[:,:,1]
    output_fft[:,:,1] = input_fft[:,:,1] * filter_fft[:,:,0] + input_fft[:,:,0] * filter_fft[:,:,1]

    output_ifft = torch.ifft(output_fft, 1)
    output_real = output_ifft[:,:,0]
    return output_real[:, indices]

def blur(image, factor):
    meas = tf.nn.avg_pool(image,[1,1,factor,factor],strides=[1,1,factor,factor],padding='VALID', data_format='NCHW')
    return meas

def get_measurements(x_batch, A, noise_batch, hparams):
    if hparams.measurement_type == 'project':
        y_batch = x_batch + noise_batch
    elif hparams.measurement_type == 'circulant':
        full_measurements = np.real(np.fft.ifft(np.fft.fft(x_batch*hparams.sign_pattern) *np.fft.fft(A)))
        indices = hparams.train_indices
        y_batch = full_measurements[:,indices] + noise_batch

    elif hparams.measurement_type == 'superres':
        x_reshape = x_batch.reshape((-1,) + hparams.image_shape)
        x_downsample = downscale_local_mean(x_reshape, (1,1, hparams.downsample,hparams.downsample))
        y_batch = x_downsample.reshape(-1, hparams.num_measurements) + noise_batch
    else:
        y_batch = np.matmul(x_batch, A) + noise_batch
    return y_batch

def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(0)
    sess = tf.Session(config=config)
    return sess

def get_noise(hparams):
    if hparams.measurement_noise_type == 'gaussian':
        noise_batch = (hparams.noise_std/np.sqrt(hparams.num_measurements)) * np.random.randn(hparams.batch_size, hparams.num_measurements)
    elif 'bm3d' in hparams.measurement_noise_type:
        measurement_noise_type = hparams.measurement_noise_type.split('-')[-1]

        noise_var = (hparams.noise_std ** 2) / hparams.num_measurements
        noise, psd, kernel = get_experiment_noise(measurement_noise_type, noise_var, 0, (hparams.y_shape[2], hparams.y_shape[3], hparams.y_shape[1]))
        noise = noise.transpose(2,0,1)
        noise_batch = noise.reshape((1,-1)).repeat(hparams.batch_size,0)
    return noise_batch

