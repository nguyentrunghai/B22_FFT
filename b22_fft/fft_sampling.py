"""
sample rotational and translational dof of one molecule relative to another
"""

from __future__ import print_function

import numpy as np
import netCDF4 as nc

from grids import PotentialGrid, ChargeGrid
from netcdf4 import write_to_nc

class FFTSampling(object):
    """
    run fft sampling for one potential grid and many charge grids
    """
    def __init__(self,
                 pot_prmtop, pot_inpcrd, pot_grid_nc,
                 char_prmtop, char_inpcrd, conf_ensenmble_for_char_grids,
                 output_nc,
                 lj_sigma_scaling_factor=1.,
                 lj_depth_scaling_factor=1.,
                 where_to_place_molecule_for_char_grids="lower_corner"):
        """
        :param pot_prmtop: str, amber prmtop file for potential grid
        :param pot_inpcrd: str, amber inpcrd file for potential grid
        :param pot_grid_nc: str, netCDF4 file storing precomputed potential grid
        :param char_prmtop: str, amber prmtop file for charge grid
        :param char_inpcrd: str, amber inpcrd for char grid
        :param conf_ensenmble_for_char_grids: 3d arrays of float, shape = (nconfs, natoms, 3)
        :param lj_sigma_scaling_factor: float
        :param lj_depth_scaling_factor: float
        :param where_to_place_molecule_for_char_grids: str
        """

        self._pot_grid = PotentialGrid(pot_prmtop, pot_inpcrd, pot_grid_nc, new_calculation=False)
        self._pot_crd = self._pot_grid.get_crd()
        self._pot_initial_com = self._pot_grid.get_initial_com()

        self._char_grid = ChargeGrid(char_prmtop, char_inpcrd, self._pot_grid,
                                     lj_sigma_scaling_factor=lj_sigma_scaling_factor,
                                     lj_depth_scaling_factor=lj_depth_scaling_factor,
                                     where_to_place_molecule=where_to_place_molecule_for_char_grids)

        self._conf_ensenmble_for_char_grids = self._load_ligand_coor_ensemble(conf_ensenmble_for_char_grids)

        self._nc_handle = nc.Dataset(output_nc, "w", format="NETCDF4")
        self._write_grid_info_to_nc()
        self._write_to_nc("pot_crd", self._pot_crd)
        self._write_to_nc("pot_initial_com", self._pot_initial_com)

    def _load_ligand_coor_ensemble(self, conf_ensenmble_for_char_grids):
        ensemble = np.asanyarray(conf_ensenmble_for_char_grids)

        assert ensemble.ndim == 3, "conf_ensenmble_for_char_grids must be 3-D array."
        natoms = self._char_grid.get_natoms()
        if (ensemble.shape[1] != natoms) or (ensemble.shape[2] != 3):
            raise RuntimeError("conf_ensenmble_for_char_grids does not have corect shape")

        return ensemble

    def _write_to_nc(self, key, value):
        if key in self._nc_handle.variables.keys():
            raise RuntimeError(key + " exists in the nc file.")
        write_to_nc(self._nc_handle, key, value)
        return None

    def _write_grid_info_to_nc(self):
        """
        write x, y, z ... to nc
        :return: None
        """
        char_grid = self._char_grid.get_grids()
        entry_names = [name for name in char_grid.keys() if name not in self._char_grid.get_grid_func_names()]
        for name in entry_names:
            self._write_to_nc(name, char_grid[name])
        return None

    def _do_fft(self, coord_for_char_grid):
        self._char_grid.cal_grids(coord_for_char_grid)
        energies = self._char_grid.get_meaningful_energies()

        for key in energies:
            print("{}: min: {}, max: {}, mean: {}".format(key, energies[key].min(),  energies[key].max(),
                                                          energies[key].mean()))

        return None

    def run_sampling(self):
        """
        :return: None
        """
        nsteps = self._conf_ensenmble_for_char_grids.shape[0]
        for step in range(nsteps):
            conf = self._conf_ensenmble_for_char_grids[step]
            self._do_fft(conf)

            key_suffix = "_%d"%step

            energies = self._char_grid.get_meaningful_energies()
            #for key in energies:
            #    self._write_to_nc(key + key_suffix, energies[key])

            self._write_to_nc("electrostatic" + key_suffix, energies["electrostatic"])
            # combine both repulsive and attractive LJ terms
            self._write_to_nc("LJ_RA" + key_suffix, energies["LJr"] + energies["LJa"])

            crd = self._char_grid.get_crd()
            self._write_to_nc("char_crd" + key_suffix, crd)

            initial_com = self._char_grid.get_initial_com()
            self._write_to_nc("char_initial_com" + key_suffix, initial_com)

            trans_corners = self._char_grid.get_meaningful_corners()
            self._write_to_nc("char_trans_corners" + key_suffix, trans_corners)

        print("Done with fft sampling, closing the nc file.")
        self._nc_handle.close()
        return None

    def get_pot_grid(self):
        return self._pot_grid

    def get_char_grid(self):
        return self._char_grid


