"""
run potential grid calculation
"""
from __future__ import print_function

import argparse

import sys
sys.path.append("/home/nguyen76/opt/src/B22_FFT/b22_fft")
from grids import PotentialGrid

parser = argparse.ArgumentParser()

parser.add_argument( "--prmtop_file", type=str, default="protein.prmtop")
parser.add_argument( "--inpcrd_file", type=str, default="protein.inpcrd")
parser.add_argument( "--grid_nc",     type=str, default="grid.nc")

parser.add_argument( "--lj_sigma_scaling_factor",     type=float, default=1.0)
parser.add_argument( "--lj_depth_scaling_factor",     type=float, default=1.0)

parser.add_argument( "--ionic_strength",     type=float, default=0.)
parser.add_argument( "--dielectric",     type=float, default=80.)

parser.add_argument( "--temperature",     type=float, default=300.)

parser.add_argument( "--spacing",     type=float, default=0.5)
parser.add_argument( "--x_count",     type=int, default=100)
parser.add_argument( "--y_count",     type=int, default=100)
parser.add_argument( "--z_count",     type=int, default=100)

parser.add_argument( "--where_to_place_molecule",     type=str, default="center")  # center or lower_corner

parser.add_argument( "--pdb_out",     type=str, default="protein.pdb")
parser.add_argument( "--box_out",     type=str, default="box.pdb")

args = parser.parse_args()

counts = (args.x_count, args.y_count, args.z_count)

pot_grid = PotentialGrid(args.prmtop_file, args.inpcrd_file, args.grid_nc,
                         new_calculation=True,
                         lj_sigma_scaling_factor=args.lj_sigma_scaling_factor,
                         lj_depth_scaling_factor=args.lj_depth_scaling_factor,
                         ionic_strength=args.ionic_strength,
                         dielectric=args.dielectric,
                         temperature=args.temperature,
                         spacing=args.spacing,
                         counts=counts,
                         where_to_place_molecule=args.where_to_place_molecule)

pot_grid.write_pdb(args.pdb_out, "w")
pot_grid.write_box(args.box_out)
print("DONE")
