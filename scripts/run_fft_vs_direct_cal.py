"""
Compare interaction energies obtained by FFT and by direct calculation
"""

from __future__ import print_function

import argparse
import pickle
import copy

import numpy as np
import netCDF4 as nc

import sys
sys.path.append("/home/nguyen76/opt/src/B22_FFT/b22_fft")
from fft_sampling import FFTSampling
from _pdb import write_pdb

parser = argparse.ArgumentParser()

parser.add_argument( "--pot_prmtop_file",   type=str, default="/home/nguyen76/B22/Leap/protein_ph9p4.prmtop")
parser.add_argument( "--pot_inpcrd_file",   type=str, default="/home/nguyen76/B22/md/protein_1/traj_0.inpcrd")
parser.add_argument( "--pot_grid_nc_file",  type=str, default="/home/nguyen76/B22/test/pot_grid/0.1M/grid.nc")

parser.add_argument( "--char_prmtop_file",   type=str, default="/home/nguyen76/B22/Leap/protein_ph9p4.prmtop")

parser.add_argument( "--char_conf_ensemble_file",   type=str, default="/home/nguyen76/B22/md/protein_2/traj.nc")
parser.add_argument( "--conf_ind",   type=int, default=0)

parser.add_argument( "--fft_sample_nc",   type=str, default="fft_samples.nc")

parser.add_argument( "--n_energy_samples",   type=int, default=10)

parser.add_argument( "--out",   type=str, default="energies.pkl")

args = parser.parse_args()


def _translate_crd(crd, trans_corner, grid_x, grid_y, grid_z):
    trans_crd = copy.deepcopy(crd)
    i, j, k = trans_corner
    x = grid_x[i]
    y = grid_y[j]
    z = grid_z[k]

    displacement = np.array([x, y, z], dtype=float)

    for atom_ix in range(len(trans_crd)):
        trans_crd[atom_ix] += displacement
    return trans_crd


char_conf = nc.Dataset(args.char_conf_ensemble_file, "r").variables["positions"][args.conf_ind : args.conf_ind+1]

sampler = FFTSampling(args.pot_prmtop_file,
                      args.pot_inpcrd_file,
                      args.pot_grid_nc_file,
                      args.char_prmtop_file,
                      char_conf,
                      args.fft_sample_nc,
                      )

sampler.run_sampling()


pot_grid = sampler.get_pot_grid()
print("COM of the the molecule used to map potential grid: {}".format(pot_grid.get_initial_com()))
pot_grid.write_box("box.pdb")
pot_grid.write_pdb("pot_molecule_initial_coord.pdb", "w")

grid_x = pot_grid.get_grids()["x"]
grid_y = pot_grid.get_grids()["y"]
grid_z = pot_grid.get_grids()["z"]

char_grid = sampler.get_char_grid()
print("COM of the the molecule used to map charge grid: {}".format(char_grid.get_initial_com()))

char_charges = char_grid.get_charges()


fft_data = nc.Dataset(args.fft_sample_nc, "r")

pot_initial_com = fft_data.variables["pot_initial_com"][:]
print("pot_initial_com: {}".format(pot_initial_com))

char_initial_com_0 = fft_data.variables["char_initial_com_0"][:]
print("char_initial_com_0: {}".format(char_initial_com_0))


char_crd_0 = fft_data.variables["char_crd_0"][:]
write_pdb(char_grid.get_prmtop(), char_crd_0, "char_molecule_initial_coord.pdb", "w")

char_crd_furthest = _translate_crd(char_crd_0, fft_data.variables["char_trans_corners_0"][-1], grid_x, grid_y, grid_z)
write_pdb(char_grid.get_prmtop(), char_crd_furthest, "char_molecule_furthest_coord.pdb", "w")

n_data_points = fft_data.variables['electrostatic_0'].shape[0]

np.random.seed(42)
sel_indices = np.random.choice(n_data_points, size=args.n_energy_samples, replace=False)


fft_energies = { "electrostatic":[], "LJa":[], "LJr":[] }
direct_energies = { "electrostatic":[], "LJa":[], "LJr":[] }


for count, ix in enumerate(sel_indices):

    fft_energies["electrostatic"].append(float(fft_data.variables['electrostatic_0'][ix]))
    fft_energies["LJr"].append(float(fft_data.variables['LJr_0'][ix]))
    fft_energies["LJa"].append(float(fft_data.variables['LJa_0'][ix]))

    corner = fft_data.variables['char_trans_corners_0'][ix]
    moved_char_crd = _translate_crd(char_crd_0, corner, grid_x, grid_y, grid_z)

    mode = "a"
    if ix == 0:
        mode = "w"
    write_pdb(char_grid.get_prmtop(), moved_char_crd, "char_molecule_move_around.pdb", mode)

    print("Calculating for conf {}".format(ix))

    de = pot_grid.direct_energy(moved_char_crd, char_charges)
    direct_energies["electrostatic"].append(de["electrostatic"])
    direct_energies["LJr"].append(de["LJr"])
    direct_energies["LJa"].append(de["LJa"])

    print("electrostatic: {} vs {}".format(fft_energies["electrostatic"][count], direct_energies["electrostatic"][count]))
    print("LJr:           {} vs {}".format(fft_energies["LJr"][count], direct_energies["LJr"][count]))
    print("LJa:           {} vs {}".format(fft_energies["LJa"][count], direct_energies["LJa"][count]))
    print(" ")

pickle.dump({"fft":fft_energies, "direct":direct_energies}, open(args.out, "w"))
