"""Inputs for celebA dataset"""

import glob
import numpy as np
from torchvision import transforms, datasets
import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib import models as nvp_model

def get_full_input(hparams):
    """Create input tensors"""
    if hparams.nc == 1:
        trans = transforms.Compose([transforms.Grayscale(), transforms.Resize((hparams.image_size,hparams.image_size)),transforms.ToTensor()])
    elif hparams.nc == 3:
        trans = transforms.Compose([transforms.Resize((hparams.image_size,hparams.image_size)),transforms.ToTensor()])
    else:
        raise NotImplementedError
    dataset = datasets.ImageFolder(hparams.input_path, transform=trans)
    if hparams.input_type == 'full-input':
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,drop_last=False,shuffle=False)
    elif hparams.input_type == 'random-test':
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,drop_last=False,shuffle=True)
    else:
        raise NotImplementedError

    dataiter = iter(dataloader)
    images = {i: next(dataiter)[0].view(-1).numpy() for i in range(hparams.num_input_images)}

    return images


def sample_generator_images(hparams):
    """Sample random images from the generator"""

    if hparams.model_name == 'realnvp':
        return sample_realnvp_images(hparams)
    else:
        raise NotImplementedError

def sample_realnvp_images(hparams):
    model = nvp_model.get_model(model_dir=os.path.dirname(hparams.checkpoint_path))
    model = model.eval()
    model = model.cuda()
    z = model.sample_z(n=hparams.batch_size)
    x = model.postprocess(model.inverse(z))
    x = x.detach().cpu().numpy()

    images = {i: image.reshape(1,-1) for (i, image) in enumerate(x)}
    return images


def model_input(hparams):
    """Create input tensors"""

    if hparams.input_type in ['full-input', 'random-test']:
        images = get_full_input(hparams)
    elif hparams.input_type == 'gen-span':
        images = sample_generator_images(hparams)
    else:
        raise NotImplementedError

    return images
