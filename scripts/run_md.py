"""
run md simulation using OpenMM
"""
from __future__ import print_function

import sys
import argparse

sys.path.append("/home/nguyen76/opt/src/B22_FFT/b22_fft")
from md_openmm import OpenMM_MD

parser = argparse.ArgumentParser()
parser.add_argument( "--prmtop_file",     type=str, default="protein.prmtop")
parser.add_argument( "--inpcrd_file",     type=str, default="protein.inpcrd")

parser.add_argument( "--phase",                     type=str, default = "OpenMM_OBC2")

parser.add_argument( "--steps_per_iteration",       type=int, default = 500)
parser.add_argument( "--niterations",               type=int, default = 1000)

parser.add_argument( "--temperature",           type=float, default = 300.)

parser.add_argument( "--nc_out",           type=str, default = "traj.nc")
parser.add_argument( "--inpcrd_prefix",    type=str, default = "none")
parser.add_argument( "--pdb_out",          type=str, default = "none")

args = parser.parse_args()

md = OpenMM_MD(args.prmtop_file, args.inpcrd_file, phase=args.phase, temperature=args.temperature)

inpcrd_prefix = args.inpcrd_prefix
if inpcrd_prefix.lower() == "none":
       inpcrd_prefix = None

pdb_out = args.pdb_out
if pdb_out.lower() == "none":
       pdb_out = None

md.run(args.nc_out, steps_per_iteration=args.steps_per_iteration, niterations=args.niterations,
       inpcrd_prefix=inpcrd_prefix, pdb_out=pdb_out)

print("Done")
