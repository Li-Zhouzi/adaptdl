import collections
import glob
import math
import os
import pandas
import functools
import pdb

from scipy.interpolate import interp1d, LinearNDInterpolator


class Application(object):
    def __init__(self,
                 name,
                 init_batch_size=None, max_batch_size=None,
                 min_local_bsz=None, max_local_bsz=None,
                 max_epochs=None, gradient_accumulation=False,
                 num_stages=1, dataset_size=None):
        self.name = name
        self.init_batch_size = init_batch_size 
        self.max_batch_size = max_batch_size 
        self.min_local_bsz = min_local_bsz 
        self.max_local_bsz = max_local_bsz 
        assert self.max_batch_size >= self.min_local_bsz
        self.max_epochs = max_epochs 
        self.gradient_accumulation = gradient_accumulation
        self.dataset_size = dataset_size
        if self.name == "cifar10":
            self.rescale_time = 120
        elif self.name == "deepspeech2":
            self.rescale_time = 150
        elif self.name == "bert":
            self.rescale_time = 300
        elif self.name == "yolov3":
            self.rescale_time = 80
        elif self.name == "imagenet":
            self.rescale_time = 250
        elif self.name == "ncf":
            self.rescale_time = 15
        else:
            self.rescale_time = 30

APPLICATIONS = {
    "bert": Application("bert", init_batch_size=4, max_batch_size=384, min_local_bsz=4, max_local_bsz=12, max_epochs=2, gradient_accumulation=True, dataset_size=97077),
    "cifar10": Application("cifar10", init_batch_size=128, max_batch_size=4096, min_local_bsz=32, max_local_bsz=1024, max_epochs=100, gradient_accumulation=True, dataset_size=50000),
    "ncf": Application("ncf", init_batch_size=256, max_batch_size=32768, min_local_bsz=32, max_local_bsz=32768, max_epochs=10, gradient_accumulation=True, dataset_size=1000000),
    "imagenet": Application("imagenet", init_batch_size=20, max_batch_size=12800, min_local_bsz=20, max_local_bsz=200, max_epochs=90, gradient_accumulation=True, dataset_size=1281167),
    "deepspeech2": Application("deepspeech2", init_batch_size=20, max_batch_size=640, min_local_bsz=10, max_local_bsz=80, max_epochs=80, gradient_accumulation=True, dataset_size=4074),
    "yolov3": Application("yolov3", init_batch_size=4, max_batch_size=512, min_local_bsz=4, max_local_bsz=8, max_epochs=50, gradient_accumulation=True, dataset_size=14041)
}

NUM_GPU_PER_NODE = 1
ARRIVAL_RATE = {
    "cifar10": 0.0024671052631578946,
    "deepspeech2": 0.0004485645933014354,
    "bert": 0.00026166267942583733,
    "imagenet": 0.0,
    "ncf": 0.0,
    "yolov3": 0.0,
}

# need perf_params and grad_params, size for each app and epoch