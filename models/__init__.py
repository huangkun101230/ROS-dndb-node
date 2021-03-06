from .shallow_partial import *
from .AE import *

import sys

def get_model(model_params):
    return PartialNet(model_params['width'], model_params['height'],\
        model_params['ndf'], model_params['dilation'],\
        model_params['norm_type'],\
        model_params['upsample_type'])

# def get_model(model_params):
#     return Generator(model_params['width'], model_params['height'],\
#         model_params['ndf'], model_params['dilation'],\
#         model_params['norm_type'],\
#         model_params['upsample_type'])