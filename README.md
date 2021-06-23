# Instance-Optimal Compressed Sensing via Posterior Sampling & Fairness for Image Generation with Uncertain Sensitive Attributes

This repo contains code for our papers [Instance-Optimal Compressed Sensing via Posterior Sampling](https://arxiv.org/abs/2106.11438) & [Fairness for Image Generation with Uncertain Sensitive Attributes]()

NOTE: Please run **all** commands from the root directory of the repository, i.e from ```code-cs-fairness/```

## Preliminaries 

1. Clone repo and install dependencies

```shell
$ git clone git@github.com:ajiljalal/code-cs-fairness.git
$ cd code-cs-fairness
$ python3.6 -m venv env
$ source env/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
$ git submodule update --init --recursive
```

2. Download data, checkpoints, and setup validation images
```shell
$ bash download.sh
$ bash shuffle_catdog.sh
```

## Reproducing quantitative results
The scripts for compressed sensing results are in ```scripts/compressed-sensing```, and the scripts for fairness are in ```scripts/fairness```.
Please adjust the command line arguments according to your requirements. ```--num-input-images``` and ```--batch-size``` need to be adjusted according to your compute capabilities and requirements.

## Visualizing results
The files ```src/view_estimated_celebA_cs.py```, ```src/view_estimated_ffhq_cs.py``` will plot qualitative reconstructions for compressed sensing. The Jupyter notebook ```src/cs_metrics.ipynb``` will plot quantitative metrics.

A similar notebook for fairness will be added shortly.

You can manually access the results under appropriately named folders in ```estimated/```.

## Citations

If you find this repo helpful, please cite the following papers:
```
@misc{jalal2021instanceoptimal,
      title={Instance-Optimal Compressed Sensing via Posterior Sampling}, 
      author={Ajil Jalal and Sushrut Karmalkar and Alexandros G. Dimakis and Eric Price},
      year={2021},
      eprint={2106.11438},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Our work uses data, code, and models from the following prior work, which must be cited according to what you use:
```
@inproceedings{song2020improved,
  author    = {Yang Song and Stefano Ermon},
  editor    = {Hugo Larochelle and
               Marc'Aurelio Ranzato and
               Raia Hadsell and
               Maria{-}Florina Balcan and
               Hsuan{-}Tien Lin},
  title     = {Improved Techniques for Training Score-Based Generative Models},
  booktitle = {Advances in Neural Information Processing Systems 33: Annual Conference
               on Neural Information Processing Systems 2020, NeurIPS 2020, December
               6-12, 2020, virtual},
  year      = {2020}
}

@article{kingma2018glow,
  title={Glow: Generative flow with invertible 1x1 convolutions},
  author={Kingma, Diederik P and Dhariwal, Prafulla},
  journal={arXiv preprint arXiv:1807.03039},
  year={2018}
}

@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}

@inproceedings{choi2020starganv2,
  title={StarGAN v2: Diverse Image Synthesis for Multiple Domains},
  author={Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}

@article{heckel_deep_2018,
    author    = {Reinhard Heckel and Paul Hand},
    title     = {Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks},
    journal   = {International Conference on Learning Representations},
    year      = {2019}
}

```


## Acknowledgments
The FFHQ NCSNv2 model was obtained from the official repo: [https://github.com/ermongroup/ncsnv2](https://github.com/ermongroup/ncsnv2)

We trained StyleGAN2 models on cats and dogs via the official repo: [https://github.com/NVlabs/stylegan2-ada](https://github.com/NVlabs/stylegan2-ada) using the AFHQ dataset (Choi et al, 2020). The FFHQ StyleGAN2 model was obtained from the official repo.

We used code from ```https://github.com/rosinality/stylegan2-pytorch``` to convert Stylegan2-ADA models from Tensorflow checkpoints to PyTorch checkpoints.

The GLOW model and code is from the official repo: [https://github.com/openai/glow](https://github.com/openai/glow). Unfortunately the provided .pb file uses placeholders for the _z_ tensors, which makes them non-differentiable and the model cannot be directly used in our experiments. In order to address this issue, we used the solution found here: [https://stackoverflow.com/a/57528005](https://stackoverflow.com/a/57528005).

