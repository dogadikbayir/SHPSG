import numpy as np
import scipy.io as sio
from funcs import *
from SHPSG import *


import time
import random
from pqdm.processes import pqdm
import pymeshlab
import argparse

from pathlib import Path

#from joblib

# subdivision level
level = 5

# create icosahedron surface
vertices, faces = icosahedron()
for i in range(level):
    # subdivide surface
    vertices, faces = subdivsurf(faces, vertices)
    # remove duplicate vertices
    vertices, faces = cleanmesh(faces,vertices)

# convert from Cartesian to spherical coordinates
sph_cor = car2sph(vertices)

# Generate the particles

# Parameter group 1: Ei, Fi = 0.75, 0.9
#                 2: D2_8 = 0.2 (roundness, lower -> rounder)
#                 3: D9_15 = 0.9 (roughness)


def task(i):
    #random.seed(i*time.time())
    # particle form: elongation index, flatness index
    Ei, Fi = 0.8, 1

    # particle roundness: larger D2_8 -- > lower roundness
    D2_8 = 0.32

    # particle roughness: larger D9_15 --> higher roughness
    D9_15 = 0

    coeff = SHPSG(Ei, Fi, D2_8, D9_15, i)

    coeff = np.array(coeff)
    #print(coeff)
    #exit(0)
    obj_path = './data/' + '_'.join([str(Ei), str(Fi), str(D2_8), str(D9_15), str(i)]) + '.obj'

    sh2obj(coeff, sph_cor, vertices, faces, obj_path, scale_factor=2)

# Create the argument parser
parser = argparse.ArgumentParser()

parser.add_argument(
    '--num_proc', type=int, default=2
)
parser.add_argument(
    '--scale_factor', type=float, default=1.0
)
parser.add_argument(
    '--num_samples', type=int, default=1000
)

opt = parser.parse_args()

Path('./data_' + str(opt.num_samples) + '_' + str(opt.scale_factor)).mkdir(parents=True, exist_ok=True)

num_samples = opt.num_samples
i_s = range(num_samples)

pqdm(i_s, task, n_jobs=opt.num_proc)
#pqdm(i_s, task, n_jobs=1)

