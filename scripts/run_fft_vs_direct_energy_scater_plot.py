
from __future__ import print_function

import argparse
import pickle

import numpy as np

from _plots import scatter_plot

parser = argparse.ArgumentParser()
parser.add_argument( "--data_pickle", type=str, default = "energies.pkl" )
parser.add_argument( "--out",         type=str, default = "fft_vs_direct.pdf" )

parser.add_argument( "--xlabel",      type=str, default = "FFT interaction energy (kcal/mol)" )
parser.add_argument( "--ylabel",      type=str, default = "Direct interaction energy (kcal/mol)" )

parser.add_argument( "--text_pos",    type=list, default = [0.55, 0.1] )

args = parser.parse_args()

data = pickle.load(open(args.data_pickle, "r"))

fft_energies = np.zeros(len(data["fft"]["electrostatic"]), dtype=float)
direct_energies = np.zeros(len(data["direct"]["electrostatic"]), dtype=float)

for name in data["fft"]:
    fft_energies += np.asarray(data["fft"][name])

for name in data["direct"]:
    direct_energies += np.asarray(data["direct"][name])


scatter_plot(fft_energies, direct_energies, args.xlabel, args.ylabel, args.out,
                show_rmse=True, 
                show_R=True,
                show_regression_line=True,
                show_regression_line_eq=True,
                show_diagonal_line=False,
                markersize=5,
                text_pos=args.text_pos,
                line_styles = {"regression" : "r--", "diagonal" : "k-"})
print("Done!")
