"""Compressed sensing main script"""
# pylint: disable=C0301,C0103,C0111

from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils
import yaml


def main(hparams):

    # Set up some stuff accoring to hparams
    hparams.n_input = np.prod(hparams.image_shape)
    utils.print_hparams(hparams)

    # get inputs
    xs_dict = utils.model_input(hparams)

    # get estimator
    estimator = utils.get_estimator(hparams, hparams.model_type)

    # set up folders, etc for checkpointing 
    utils.setup_checkpointing(hparams)

    # get saved results
    measurement_losses, l2_losses, z_hats, likelihoods = utils.load_checkpoints(hparams)

    x_batch_dict = {}
    x_hats_dict = {}

    A = utils.get_A(hparams)



    for key, x in xs_dict.items():
        if not hparams.not_lazy:
            # If lazy, first check if the image has already been
            # saved before . If yes, then skip this image.
            save_path = utils.get_save_path(hparams, key)
            is_saved = os.path.isfile(save_path) 
            if is_saved:
                continue

        x_batch_dict[key] = x
        if len(x_batch_dict) < hparams.batch_size:
            continue

        # Reshape input
        x_batch_list = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.items()]
        x_batch = np.concatenate(x_batch_list)

        # Construct noise and measurements
        noise_batch = utils.get_noise(hparams)
        y_batch = utils.get_measurements(x_batch, A, noise_batch, hparams)

        # Construct estimates 
        x_hat_batch, z_hat_batch, likelihood_batch = estimator(A, y_batch, hparams)

        for i, key in enumerate(x_batch_dict.keys()):
            x = xs_dict[key]
            y_train = y_batch[i]
            x_hat = x_hat_batch[i]

            # Save the estimate
            x_hats_dict[key] = x_hat

            # Compute and store measurement and l2 loss
            measurement_losses[key] = utils.get_measurement_loss(x_hat, A, y_train, hparams)
            l2_losses[key] = utils.get_l2_loss(x_hat, x)
            z_hats[key] = z_hat_batch[i]
            likelihoods[key] = likelihood_batch[i]

        print('Processed upto image {0} / {1}'.format(key+1, len(xs_dict)))

        # Checkpointing
        if (not hparams.debug) and ((key+1) % hparams.checkpoint_iter == 0):
            utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, z_hats, likelihoods, hparams)
            x_hats_dict = {} 
            print('\nProcessed and saved first ', key+1, 'images\n')

        x_batch_dict = {}

    # Final checkpoint
    if not hparams.debug:
        utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, z_hats, likelihoods, hparams)
        print('\nProcessed and saved all {0} image(s)\n'.format(len(xs_dict)))

    if hparams.print_stats:
        print(hparams.model_type)
        measurement_loss_list = list(measurement_losses.values())
        l2_loss_list = list(l2_losses.values())
        mean_m_loss = np.mean(measurement_loss_list)
        mean_l2_loss = np.mean(l2_loss_list)
        print('mean measurement loss = {0}'.format(mean_m_loss))
        print('mean l2 loss = {0}'.format(mean_l2_loss))

    if hparams.image_matrix > 0:
        utils.image_matrix(xs_dict, x_hats_dict, hparams)

    # Warn the user that some things were not processsed
    if len(x_batch_dict) > 0:
        print('\nDid NOT process last {} images because they did not fill up the last batch.'.format(len(x_batch_dict)))
        print('Consider rerunning lazily with a smaller batch size.')

if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--checkpoint-path', type=str, default='models/ncsnv2_ffhq/checkpoint_80000.pth', help='Path to pretrained model')
    PARSER.add_argument('--net', type=str, default='ncsnv2', help='Name of model. options = [glow, stylegan2, ncsnv2, dd]')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--image-size', type=int, default=256, help='size of image')
    PARSER.add_argument('--input-type', type=str, default='full-input', help='Where to take input from')
    PARSER.add_argument('--num-input-images', type=int, default=2, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=2, help='How many examples are processed together')
    PARSER.add_argument('--cache-dir', type=str, default='cache', help='cache directory for model weights')
    PARSER.add_argument('--ncsnv2-configs-file', type=str, default='./ncsnv2/configs/ffhq.yml', help='location of ncsnv2 config file')


    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type')
    PARSER.add_argument('--noise-std', type=float, default=1, help='expected norm of noise')
    PARSER.add_argument('--measurement-noise-type', type=str, default='gaussian', help='type of noise')

    # Measurement type specific hparams
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')
    PARSER.add_argument('--downsample', type=int, default=None, help='downsampling factor')

    # Model
    PARSER.add_argument('--model-type', type=str, default=None, required=True, help='model used for estimation. options=[map, langevin, pulse, dd]')
    PARSER.add_argument('--mloss-weight', type=float, default=-1, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior-weight', type=float, default=-1, help='weight on z prior')
    PARSER.add_argument('--zprior-sdev', type=float, default=1.0, help='standard deviation for target distributon of  z')
    PARSER.add_argument('--zprior-init-sdev', type=float, default=1.0, help='standard deviation to initialize z')
    PARSER.add_argument('--T', type=float, default=-1, help='number of iterations for each level of noise in Langevin annealing')
    PARSER.add_argument('--L', type=float, default=-1, help='number of noise levels for annealing Langevin')
    PARSER.add_argument('--sigma-init', type=float, default=-1, help='initial noise level for annealing langevin')
    PARSER.add_argument('--sigma-final', type=float, default=-1, help='final noise level for annealing Langevin')
    PARSER.add_argument('--error-threshold', type=float, default=0., help='threshold for measurement error before restart')
    PARSER.add_argument('--num-noise-variables', type=int, default=5, help='STYLEGAN2 : number of noise variables in  to optimize')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='sgd', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.4, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0., help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=1000, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=1, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')

    #PULSE arguments
    PARSER.add_argument('--seed', type=int, help='manual seed to use')
    PARSER.add_argument('--loss-str', type=str, default="100*L2+0.05*GEOCROSS", help='Loss function to use')
    PARSER.add_argument('--pulse-eps', type=float, default=2e-3, help='Target for downscaling loss (L2)')
    PARSER.add_argument('--noise-type', type=str, default='trainable', help='zero, fixed, or trainable')
    PARSER.add_argument('--tile-latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
    PARSER.add_argument('--lr-schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')

    # Output
    PARSER.add_argument('--not-lazy', action='store_true', help='whether the evaluation is lazy')
    PARSER.add_argument('--debug', action='store_true', help='debug mode does not save images or stats')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print statistics')
    PARSER.add_argument('--checkpoint-iter', type=int, default=50, help='checkpoint every x batches')
    PARSER.add_argument('--image-matrix', type=int, default=0,
                        help='''
                                0 = 00 =      no       image matrix,
                                1 = 01 =          show image matrix
                                2 = 10 = save          image matrix
                                3 = 11 = save and show image matrix
                             '''
                       )
    PARSER.add_argument('--gif', action='store_true', help='whether to create a gif')
    PARSER.add_argument('--gif-iter', type=int, default=1, help='save gif frame every x iter')
    PARSER.add_argument('--gif-dir', type=str, default='', help='where to store gif frames')

    PARSER.add_argument('--cuda', dest='cuda', action='store_true')
    PARSER.add_argument('--no-cuda', dest='cuda', action='store_false')
    PARSER.set_defaults(cuda=True)

    PARSER.add_argument('--project', dest='project', action='store_true')
    PARSER.add_argument('--no-project', dest='project', action='store_false')
    PARSER.set_defaults(project=True)

    PARSER.add_argument('--annealed', dest='annealed', action='store_true')
    PARSER.add_argument('--no-annealed', dest='annealed', action='store_false')
    PARSER.set_defaults(annealed=False)

    HPARAMS = PARSER.parse_args()
    HPARAMS.input_path = f'./test_images/{HPARAMS.dataset}'
    if HPARAMS.cuda:
        HPARAMS.device='cuda:0'
    else:
        HPARAMS.device = 'cpu:0'


    if HPARAMS.net == 'ncsnv2':
        with open(HPARAMS.ncsnv2_configs_file, 'r') as f:
            HPARAMS.ncsnv2_configs = yaml.load(f)
        HPARAMS.ncsnv2_configs['sampling']['step_lr'] = HPARAMS.learning_rate
        HPARAMS.ncsnv2_configs['sampling']['n_steps_each'] = int(HPARAMS.T)
        HPARAMS.ncsnv2_configs['model']['sigma_begin'] = int(HPARAMS.sigma_init)
        HPARAMS.ncsnv2_configs['model']['sigma_end'] = HPARAMS.sigma_final

    HPARAMS.image_shape = (3, HPARAMS.image_size, HPARAMS.image_size)
    HPARAMS.n_input = np.prod(HPARAMS.image_shape)

    if HPARAMS.measurement_type == 'circulant':
        HPARAMS.train_indices = np.random.randint(0, HPARAMS.n_input, HPARAMS.num_measurements )
        HPARAMS.sign_pattern = np.float32((np.random.rand(1,HPARAMS.n_input) < 0.5)*2 - 1.)
    elif HPARAMS.measurement_type == 'superres':
        HPARAMS.y_shape = (HPARAMS.batch_size, HPARAMS.image_shape[0], HPARAMS.image_size//HPARAMS.downsample,HPARAMS.image_size//HPARAMS.downsample)
        HPARAMS.num_measurements = np.prod(HPARAMS.y_shape[1:])
    elif HPARAMS.measurement_type == 'project':
        HPARAMS.y_shape = (HPARAMS.batch_size, HPARAMS.image_shape[0], HPARAMS.image_size, HPARAMS.image_size)
        HPARAMS.num_measurements = np.prod(HPARAMS.y_shape[1:])


    from utils import view_image 

    if HPARAMS.mloss_weight < 0:
        HPARAMS.mloss_weight = None
    if HPARAMS.zprior_weight < 0:
        HPARAMS.zprior_weight = None
    if HPARAMS.annealed:
        if HPARAMS.T < 0:
            HPARAMS.T = 200
        if HPARAMS.L < 0:
            HPARAMS.L = 10
        if HPARAMS.sigma_final < 0:
            HPARAMS.sigma_final = HPARAMS.noise_std
        if HPARAMS.sigma_init < 0:
            HPARAMS.sigma_init =  100 * HPARAMS.sigma_final
        HPARAMS.max_update_iter = int(HPARAMS.T * HPARAMS.L)

    main(HPARAMS)


