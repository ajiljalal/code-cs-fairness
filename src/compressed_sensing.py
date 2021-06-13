"""Compressed sensing main script"""
# pylint: disable=C0301,C0103,C0111

from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils
from lpips import PerceptualLoss
import yaml


def main(hparams):
    # set up perceptual loss
    device = 'cuda:0'
    percept = PerceptualLoss(
	    model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    # Set up some stuff accoring to hparams
    hparams.n_input = np.prod(hparams.image_shape)
    #utils.set_num_measurements(hparams)
    utils.print_hparams(hparams)

    # get inputs
    xs_dict = model_input(hparams)

    estimators = utils.get_estimators(hparams)
    utils.setup_checkpointing(hparams)
    measurement_losses, l2_losses, lpips_scores, z_hats, likelihoods = utils.load_checkpoints(hparams)

    x_hats_dict = {model_type : {} for model_type in hparams.model_types}
    x_batch_dict = {}

    A = utils.get_A(hparams)



    for key, x in xs_dict.items():
        if not hparams.not_lazy:
            # If lazy, first check if the image has already been
            # saved before by *all* estimators. If yes, then skip this image.
            save_paths = utils.get_save_paths(hparams, key)
            is_saved = all([os.path.isfile(save_path) for save_path in save_paths.values()])
            if is_saved:
                continue

        x_batch_dict[key] = x
        if len(x_batch_dict) < hparams.batch_size:
            continue

        # Reshape input
        x_batch_list = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.items()]
        x_batch = np.concatenate(x_batch_list)

        # Construct noise and measurements

        if hparams.fixed_init:
            noise = (hparams.noise_std/np.sqrt(hparams.num_measurements)) * np.random.randn(1, hparams.num_measurements)
            noise_batch = noise.repeat(hparams.batch_size,0)
        else:

            #noise_batch = (hparams.noise_std/np.sqrt(hparams.num_measurements)) * np.random.randn(hparams.batch_size, hparams.num_measurements)
            noise_batch = utils.get_noise(hparams)
            print(noise_batch.shape)

        y_batch = utils.get_measurements(x_batch, A, noise_batch, hparams)

        # Construct estimates using each estimator
        for model_type in hparams.model_types:
            estimator = estimators[model_type]
            x_hat_batch, z_hat_batch, likelihood_batch = estimator(A, y_batch, hparams)

            for i, key in enumerate(x_batch_dict.keys()):
                x = xs_dict[key]
                y_train = y_batch[i]
                x_hat = x_hat_batch[i]

                # Save the estimate
                x_hats_dict[model_type][key] = x_hat

                # Compute and store measurement and l2 loss
                measurement_losses[model_type][key] = utils.get_measurement_loss(x_hat, A, y_train, hparams)
                l2_losses[model_type][key] = utils.get_l2_loss(x_hat, x)
                lpips_scores[model_type][key] = utils.get_lpips_score(percept, x_hat, x, hparams.image_shape)
                z_hats[model_type][key] = z_hat_batch[i]
                likelihoods[model_type][key] = likelihood_batch[i]

        print('Processed upto image {0} / {1}'.format(key+1, len(xs_dict)))

        # Checkpointing
        if (hparams.save_images) and ((key+1) % hparams.checkpoint_iter == 0):
            utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, lpips_scores, z_hats, likelihoods, save_image, hparams)
            x_hats_dict = {model_type : {} for model_type in hparams.model_types}
            print('\nProcessed and saved first ', key+1, 'images\n')

        x_batch_dict = {}

    # Final checkpoint
    if hparams.save_images:
        utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, lpips_scores, z_hats, likelihoods, save_image, hparams)
        print('\nProcessed and saved all {0} image(s)\n'.format(len(xs_dict)))

    if hparams.print_stats:
        for model_type in hparams.model_types:
            print(model_type)
            measurement_loss_list = list(measurement_losses[model_type].values())
            l2_loss_list = list(l2_losses[model_type].values())
            mean_m_loss = np.mean(measurement_loss_list)
            mean_l2_loss = np.mean(l2_loss_list)
            print('mean measurement loss = {0}'.format(mean_m_loss))
            print('mean l2 loss = {0}'.format(mean_l2_loss))

    if hparams.image_matrix > 0:
        utils.image_matrix(xs_dict, x_hats_dict, view_image, hparams)

    # Warn the user that some things were not processsed
    if len(x_batch_dict) > 0:
        print('\nDid NOT process last {} images because they did not fill up the last batch.'.format(len(x_batch_dict)))
        print('Consider rerunning lazily with a smaller batch size.')

if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--checkpoint-path', type=str, default='models/realnvp_celebahq/ckpt_ep300.pt', help='Path to pretrained model')
    PARSER.add_argument('--net', type=str, default='realnvp', help='Name of model. options = [realnvp, glow]')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--input-type', type=str, default='full-input', help='Where to take input from')
    PARSER.add_argument('--num-input-images', type=int, default=2, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=2, help='How many examples are processed together')
    PARSER.add_argument('--image-size', type=int, default=64, help='size of image')
    PARSER.add_argument('--nc', type=int, default=3, help='number of color channels')
    PARSER.add_argument('--complex', action='store_true', help='specify that the expected image is complex valued')
    PARSER.add_argument('--cache-dir', type=str, default='cache', help='cache directory for model weights')
    PARSER.add_argument('--ncsnv2-configs-file', type=str, default='./ncsnv2/configs/ffhq.yml', help='location of ncsnv2 config file')


    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type')
    PARSER.add_argument('--noise-std', type=float, default=1, help='std dev of noise')
    PARSER.add_argument('--measurement-noise-type', type=str, default='gaussian', help='type of noise')

    # Measurement type specific hparams
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')
    PARSER.add_argument('--downsample', type=int, default=None, help='downsampling factor')

    # Model
    PARSER.add_argument('--model-types', type=str, nargs='+', default=None, help='model(s) used for estimation')
    PARSER.add_argument('--mloss-weight', type=float, default=-1, help='L2 measurement loss weight')
    PARSER.add_argument('--ploss-weight', type=float, default=-1, help='log(p(G(z)))')
    PARSER.add_argument('--zprior-weight', type=float, default=-1, help='weight on z prior')
    PARSER.add_argument('--zprior-sdev', type=float, default=1.0, help='standard deviation for target distributon of  z')
    PARSER.add_argument('--zprior-init-sdev', type=float, default=1.0, help='standard deviation to initialize z')
    PARSER.add_argument('--T', type=float, default=-1, help='number of iterations for each level of noise in Langevin annealing')
    PARSER.add_argument('--L', type=float, default=-1, help='number of noise levels for annealing Langevin')
    PARSER.add_argument('--sigma-init', type=float, default=-1, help='initial noise level for annealing langevin')
    PARSER.add_argument('--sigma-final', type=float, default=-1, help='final noise level for annealing Langevin')
    PARSER.add_argument('--gradient-noise-weight', type=float, default=-1., help='norm of noise to add in gradient')
    PARSER.add_argument('--error-threshold', type=float, default=0., help='threshold for measurement error before restart')
    PARSER.add_argument('--num-noise-variables', type=int, default=5, help='STYLEGAN2 : number of noise variables in  to optimize')
    PARSER.add_argument('--fixed-init', action='store_true', help='whether initialization should be from a fixed point')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='adam', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.4, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=1000, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=10, help='number of random restarts')
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
    PARSER.add_argument('--save-images', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--save-stats', action='store_true', help='whether to save estimated images')
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

    PARSER.add_argument('--lpips', dest='lpips', action='store_true')
    PARSER.add_argument('--no-lpips', dest='lpips', action='store_false')
    PARSER.set_defaults(lpips=False)


    HPARAMS = PARSER.parse_args()
    HPARAMS.input_path = f'./test_images/{HPARAMS.dataset}'
    if HPARAMS.cuda:
        HPARAMS.device='cuda:0'
    else:
        HPARAMS.device = 'cpu:0'


    if HPARAMS.net == 'realnvp':
        HPARAMS.image_size = 64
    elif HPARAMS.net == 'glow':
        HPARAMS.image_size = 256
    elif HPARAMS.net == 'ncsnv2':
        # if HPARAMS.dataset == 'ffhq':
        #     HPARAMS.image_size = 256
        # elif HPARAMS.dataset == 'celebA':
        #     HPARAMS.image_size = 64
        # else:
        #     raise NotImplementedError
        print(HPARAMS.ncsnv2_configs_file)
        with open(HPARAMS.ncsnv2_configs_file, 'r') as f:
            HPARAMS.ncsnv2_configs = yaml.load(f)
        HPARAMS.ncsnv2_configs['sampling']['step_lr'] = HPARAMS.learning_rate
        HPARAMS.ncsnv2_configs['sampling']['n_steps_each'] = int(HPARAMS.T)
        HPARAMS.ncsnv2_configs['model']['sigma_begin'] = int(HPARAMS.sigma_init)
        HPARAMS.ncsnv2_configs['model']['sigma_end'] = HPARAMS.sigma_final

        if HPARAMS.nc == 3:
            HPARAMS.ncsnv2_configs['data']['channels'] = 3
        elif HPARAMS.nc == 1:
            if HPARAMS.complex:
                HPARAMS.ncsnv2_configs['data']['channels'] = 2
            else:
                HPARAMS.ncsnv2_configs['data']['channels'] = 1
        else:
            raise NotImplementedError


    HPARAMS.image_shape = (HPARAMS.nc, HPARAMS.image_size, HPARAMS.image_size)
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


    from celebA_input import model_input
    from celebA_utils import view_image, save_image

    if HPARAMS.mloss_weight < 0:
        HPARAMS.mloss_weight = None
    if HPARAMS.zprior_weight < 0:
        HPARAMS.zprior_weight = None
    if HPARAMS.ploss_weight < 0:
        HPARAMS.ploss_weight = None
    if HPARAMS.gradient_noise_weight < 0:
        HPARAMS.gradient_noise_weight = None
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


