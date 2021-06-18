# Instance-Optimal Compressed Sensing via Posterior Sampling & Fairness for Image Generation with Uncertain Sensitive Attributes

This repo contains code for our papers [Instance-Optimal Compressed Sensing via Posterior Sampling]() & [Fairness for Image Generation with Uncertain Sensitive Attributes]()

NOTE: Please run **all** commands from the root directory of the repository, i.e from ``code-cs-fairness/```

## Preliminaries 

1. Clone repo and install dependencies

```shell
$ git clone git@github.com:ajiljalal/code-cs-fairness.git
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

### Reproducing quantitative results


# Citations

If you find this repo helpful, please cite the following papers:

Our work uses data, code, and models from the following prior work, which must be cited:
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
```


# Acknowledgments
We use code from ```https://github.com/rosinality/stylegan2-pytorch``` to convert Stylegan2-ADA models from Tensorflow checkpoints to PyTorch checkpoints.
