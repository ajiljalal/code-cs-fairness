#Instance-Optimal Compressed Sensing via Posterior Sampling & Fairness for Image Generation with Uncertain Sensitive Attributes

This repo contains code for our papers [Instance-Optimal Compressed Sensing via Posterior Sampling]() & [Fairness for Image Generation with Uncertain Sensitive Attributes]()

NOTE: Please run **all** commands from the root directory of the repository, i.e from ``code-cs-fairness/```

## Preliminaries 
---

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
