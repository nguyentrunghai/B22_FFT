"""
Compare interaction energies obtained by FFT and by direct calculation
"""

from __future__ import print_function

import argparse

import numpy as np
import netCDF4 as nc

import sys
sys.path.append("/home/nguyen76/opt/src/B22_FFT/b22_fft")
from fft_sampling import FFTSampling
from _pdb import write_pdb

parser = argparse.ArgumentParser()

parser.add_argument( "--pot_prmtop_file",   type=str, default="/home/nguyen76/B22/Leap/protein_ph9p4.prmtop")
parser.add_argument( "--pot_inpcrd_file",   type=str, default="/home/nguyen76/B22/md/protein_1/traj_0.inpcrd")
parser.add_argument( "--pot_grid_nc_file",  type=str, default="/scratch/nguyen76/B22/pot_grids/0.1M/grid.nc")

parser.add_argument( "--char_prmtop_file",   type=str, default="/home/nguyen76/B22/Leap/protein_ph9p4.prmtop")
parser.add_argument( "--char_inpcrd_file",   type=str, default="/home/nguyen76/B22/md/protein_2/traj_0.inpcrd")

parser.add_argument( "--char_conf_ensemble_file",   type=str, default="/home/nguyen76/B22/md/protein_2/traj.nc")
parser.add_argument( "--conf_ind",   type=int, default=0)

parser.add_argument( "--lj_sigma_scaling_factor",   type=float, default=1.0)
parser.add_argument( "--lj_depth_scaling_factor",   type=float, default=1.0)

parser.add_argument( "--where_to_place_molecule_for_char_grids",   type=str, default="lower_corner")

parser.add_argument( "--fft_sample_nc",   type=str, default="fft_samples.nc")

parser.add_argument( "--n_energy_samples",   type=int, default=100)

args = parser.parse_args()

char_conf = nc.Dataset(args.char_conf_ensemble_file, "r").variables["positions"][args.conf_ind : args.conf_ind+1]

sampler = FFTSampling(args.pot_prmtop_file, args.pot_inpcrd_file, args.pot_grid_nc_file,
                      args.char_prmtop_file, args.char_inpcrd_file, char_conf,
                      args.fft_sample_nc,
                      lj_sigma_scaling_factor=args.lj_sigma_scaling_factor,
                      lj_depth_scaling_factor=args.lj_depth_scaling_factor,
                      where_to_place_molecule_for_char_grids=args.where_to_place_molecule_for_char_grids)

sampler.run_sampling()

pot_grid = sampler.get_pot_grid()
print("COM of the the molecule used to map potential grid: {}".format(pot_grid.get_initial_com()))
pot_grid.write_box("box.pdb")
pot_grid.write_pdb("pot_molecule_initial_coord.pdb", "w")

spacing = pot_grid.get_grids()['spacing']

char_grid = sampler.get_char_grid()
print("COM of the the molecule used to map charge grid: {}".format(char_grid.get_initial_com()))

fft_data = nc.Dataset(args.fft_sample_nc, "r")

pot_initial_com = fft_data.variables["pot_initial_com"][:]
print("pot_initial_com: {}".format(pot_initial_com))

char_initial_com_0 = fft_data.variables["char_initial_com_0"][:]
print("char_initial_com_0: {}".format(char_initial_com_0))

char_crd_0 = fft_data.variables["char_crd_0"][:]
write_pdb(char_grid.get_prmtop(), char_crd_0, "char_molecule_initial_coord.pdb", "w")

n_data_points = fft_data.variables['electrostatic_0'].shape[0]
assert n_data_points == fft_data.variables['LJ_RA_0'].shape[0], "energy arrays do not have the same len"
assert n_data_points == fft_data.variables['char_trans_corners_0'].shape[0], "energy and trans corner arrays do not have the same len"

np.random.seed(42)
sel_indices = np.random.choice(n_data_points, size=args.n_energy_samples, replace=False)

fft_energies = fft_data.variables['electrostatic_0'][sel_indices] + fft_data.variables['LJ_RA_0'][sel_indices]
trans_corners = fft_data.variables['char_trans_corners_0'][sel_indices]

direct_energies = []
for i, corner in enumerate(trans_corners):
    move_by = corner * spacing
    moved_char_crd = char_crd_0 + move_by

    mode = "a"
    if i == 0:
        mode = "w"
    write_pdb(char_grid.get_prmtop(), moved_char_crd, "char_molecule_move_around.pdb", mode)


