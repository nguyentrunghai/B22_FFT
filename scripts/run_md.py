"""
run md simulation using OpenMM
"""
from __future__ import print_function

import os
import sys
import argparse

sys.path.append("../b22_fft")
from md_openmm import OpenMM_MD

parser = argparse.ArgumentParser()
parser.add_argument( "--prmtop_file",     type=str, default="protein.prmtop")
parser.add_argument( "--inpcrd_file",     type=str, default="protein.inpcrd")

parser.add_argument( "--phase",                     type=str, default = "OpenMM_OBC2")

parser.add_argument( "--steps_per_iteration",       type=int, default = 500)
parser.add_argument( "--niterations",               type=int, default = 1000)
parser.add_argument( "--rotations_per_iteration",   type=int, default = 1)

parser.add_argument( "--temperature",           type=float, default = 300.)

parser.add_argument( "--out_prefix",           type=float, default = "traj")

args = parser.parse_args()

md = OpenMM_MD(args.prmtop_file, args.inpcrd_file, phase=args.phase, temperature=args.temperature)

nc_file_name = args.out_prefix + ".nc"
inpcrd_prefix = args.out_prefix
pdb_out = args.out_prefix + ".pdb"

md.run(nc_file_name, steps_per_iteration=args.steps_per_iteration, niterations=args.niterations,
       inpcrd_prefix=inpcrd_prefix, pdb_out=pdb_out)

print("Done")
