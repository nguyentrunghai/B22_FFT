"""
run fft sampling for one configuration (conformation + rotation) of protein 1 
and multiple configurations of protein 2
"""

from __future__ import print_function

import argparse

import sys
sys.path.append("/home/nguyen76/opt/src/B22_FFT/b22_fft")
from fft_sampling import FFTSampling

import netCDF4 as nc


parser = argparse.ArgumentParser()

parser.add_argument( "--pot_prmtop_file",   type=str, default="protein_1.prmtop")
parser.add_argument( "--pot_inpcrd_file",   type=str, default="protein_1.inpcrd")
parser.add_argument( "--pot_grid_nc_file",  type=str, default="grid.nc")

parser.add_argument( "--char_prmtop_file",   type=str, default="protein_2.prmtop")
parser.add_argument( "--char_inpcrd_file",   type=str, default="protein_2.inpcrd")

parser.add_argument( "--char_conf_ensemble_file",   type=str, default="traj.nc")
parser.add_argument( "--begin_conf",   type=int, default=0)
parser.add_argument( "--end_conf",     type=int, default=2)

parser.add_argument( "--lj_sigma_scaling_factor",   type=float, default=1.0)
parser.add_argument( "--lj_depth_scaling_factor",   type=float, default=1.0)

parser.add_argument( "--where_to_place_molecule_for_char_grids",   type=str, default="lower_corner")

parser.add_argument( "--nc_out",   type=str, default="fft_samples.nc")

args = parser.parse_args()

assert args.begin_conf >= 0, "begin_conf must be >= 0"

char_conf_emsemble = nc.Dataset(args.char_conf_ensemble_file, "r").variables["positions"]

assert args.end_conf <= char_conf_emsemble.shape[0], "end_conf larger than the number of confs in " + args.char_conf_ensemble_file

char_conf_emsemble = char_conf_emsemble[args.begin_conf : args.end_conf]

sampler = FFTSampling(args.pot_prmtop_file, args.pot_inpcrd_file, args.pot_grid_nc_file,
                      args.char_prmtop_file, args.char_inpcrd_file, char_conf_emsemble,
                      args.nc_out,
                      lj_sigma_scaling_factor=args.lj_sigma_scaling_factor,
                      lj_depth_scaling_factor=args.lj_depth_scaling_factor,
                      where_to_place_molecule_for_char_grids=args.where_to_place_molecule_for_char_grids)
sampler.run_sampling()

print("DONE")
