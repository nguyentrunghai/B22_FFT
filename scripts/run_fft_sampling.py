"""
run fft sampling for one configuration (conformation + rotation) of protein 1 
and multiple configurations of protein 2
"""

from __future__ import print_function

import argparse

import sys
sys.path.append("/home/nguyen76/opt/src/B22_FFT/b22_fft")
from fft_sampling import FFTSampling

parser = argparse.ArgumentParser()

parser.add_argument( "--pot_prmtop_file",   type=str, default="protein_1.prmtop")
parser.add_argument( "--pot_inpcrd_file",   type=str, default="protein_1.inpcrd")
parser.add_argument( "--pot_grid_nc_file",  type=str, default="grid.nc")

parser.add_argument( "--char_prmtop_file",   type=str, default="protein_2.prmtop")
parser.add_argument( "--char_inpcrd_file",   type=str, default="protein_2.inpcrd")

parser.add_argument( "--char_conf_ensemble_file",   type=str, default="traj.nc")

args = parser.parse_args()