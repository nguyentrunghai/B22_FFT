
from __future__ import print_function

import os

import numpy as np
import netCDF4 as nc

from amber_par import AmberPrmtopLoader, InpcrdLoader
from _pdb import write_pdb, write_box
from netcdf4 import write_to_nc

from util import c_cal_charge_grid
from util import c_cal_potential_grid_electrostatic, c_cal_potential_grid_LJa, c_cal_potential_grid_LJr
from util import c_cal_potential_grid_occupancy


class Grid(object):
    """
    an base class defining some common methods and data attributes
    working implementations are in PotentialGrid and ChargeGrid below
    """
    def __init__(self):
        self._grid = {}

        self._grid_func_names = ("electrostatic", "LJr", "LJa", "occupancy")

        # keys to be stored in the result nc file
        #cartesian_axes = ("x", "y", "z")
        #box_dim_names = ("d0", "d1", "d2")
        #others = ("spacing", "counts", "origin", "lj_sigma_scaling_factor")
        #self._grid_allowed_keys = self._grid_func_names + cartesian_axes + box_dim_names + others

        # 8 corners of a cubic box
        self._eight_corner_shifts = [np.array([i,j,k], dtype=int) for i in range(2) for j in range(2) for k in range(2)]
        self._eight_corner_shifts = np.array(self._eight_corner_shifts, dtype=int)

        # 6 corners surrounding a point
        self._six_corner_shifts = self._get_six_corner_shifts()

    def _get_six_corner_shifts(self):
        six_corner_shifts = []
        for i in [-1, 1]:
            six_corner_shifts.append(np.array([i, 0, 0], dtype=int))
            six_corner_shifts.append(np.array([0, i, 0], dtype=int))
            six_corner_shifts.append(np.array([0, 0, i], dtype=int))
        return np.array(six_corner_shifts, dtype=int)
    
    def _set_grid_key_value(self, key, value):
        """
        :param key: str
        :param value: any object
        :return: None
        """
        print("Setting " + key)
        if key in self._grid:
            print(key + " exists. Overide!")

        self._grid[key] = value
        return None
    
    def _load_prmtop(self, prmtop_file_name, lj_sigma_scaling_factor, lj_depth_scaling_factor):
        """
        :param prmtop_file_name: str, name of AMBER prmtop file
        :param lj_sigma_scaling_factor: float, must have value in [0.5, 1.0].
        It is stored in self._grid["lj_sigma_scaling_factor"] as
        a array of shape (1,) for reason of saving to nc file.
        :return: None
        """
        print("Loading " + prmtop_file_name)
        assert 0.5 <= lj_sigma_scaling_factor <= 1.0, "lj_sigma_scaling_factor is out of allowed range"
        self._prmtop = AmberPrmtopLoader(prmtop_file_name).get_parm_for_grid_calculation()

        # scale the Lennard Jones sigma parameters
        self._prmtop["LJ_SIGMA"] *= lj_sigma_scaling_factor

        # scale Lennard Jones depth
        self._prmtop["A_LJ_CHARGE"] *= np.sqrt(lj_depth_scaling_factor)
        self._prmtop["R_LJ_CHARGE"] *= np.sqrt(lj_depth_scaling_factor)

        # save "lj_sigma_scaling_factor" and "lj_depth_scaling_factor" to self._grid
        self._set_grid_key_value("lj_sigma_scaling_factor", np.array([lj_sigma_scaling_factor], dtype=float))
        self._set_grid_key_value("lj_depth_scaling_factor", np.array([lj_depth_scaling_factor], dtype=float))
        return None
    
    def _load_inpcrd(self, inpcrd_file_name):
        """
        save the molecule's coordinates to self._crd
        :param inpcrd_file_name: str
        :return: None
        """
        print("Loading file: " + inpcrd_file_name)
        self._crd = InpcrdLoader(inpcrd_file_name).get_crd()

        # check if prmtop and inpcrd have the same number of atoms
        natoms = self._prmtop["POINTERS"]["NATOM"]
        if (self._crd.shape[0] != natoms) or (self._crd.shape[1] != 3):
            raise RuntimeError("Coordinates in %s has wrong shape"%inpcrd_file_name)
        return None
    
    def _move_molecule_to(self, location):
        """
        Move the center of mass of the molecule to location.
        Calling this function will modify self._crd
        :param location: 1d ndarray of shape (3,) or list with len = 3
        :return: None
        """
        assert len(location) == 3, "location must have len 3"

        displacement = np.array(location, dtype=float) - self._get_molecule_center_of_mass()
        self._crd += displacement.reshape(1, 3)
        return None
    
    def _get_molecule_center_of_mass(self):
        """
        return the center of mass of self._crd
        """
        masses = self._prmtop["MASS"]
        total_mass = masses.sum()
        if total_mass == 0:
            raise RuntimeError("Zero total mass")

        center_of_mass = (masses.reshape(-1, 1) * self._crd).sum(axis=0)
        return center_of_mass / total_mass

    def _get_corner_crd(self, corner):
        """
        map corner's indices to its geometric coordinates
        :param corner: 1d array or list of int with len = 3
        :return: 1d array of float with shape (3,)
        """
        i, j, k = corner
        return np.array([self._grid["x"][i], self._grid["y"][j], self._grid["z"][k]] , dtype=float)
    
    def _get_upper_most_corner(self):
        return np.array(self._grid["counts"] - 1, dtype=int)
    
    def _get_upper_most_corner_crd(self):
        upper_most_corner = self._get_upper_most_corner()
        return self._get_corner_crd(upper_most_corner)
    
    def _get_origin_crd(self):
        return self._get_corner_crd([0,0,0])

    def _initialize_convenient_para(self):
        self._origin_crd = self._get_origin_crd()
        self._upper_most_corner_crd = self._get_upper_most_corner_crd()
        self._upper_most_corner = self._get_upper_most_corner()
        self._spacing = np.array([self._grid["d%d"%i][i] for i in range(3)], dtype=float)
        return None

    def _is_in_grid(self, atom_coordinate):
        """
        in grid means atom_coordinate >= origin_crd and atom_coordinate < uper_most_corner_crd
        :param atom_coordinate: 3-array of float
        :return: bool
        """
        if np.any(atom_coordinate < self._origin_crd) or np.any(atom_coordinate >= self._upper_most_corner_crd):
            return False
        return True
    
    def _distance(self, corner, atom_coordinate):
        """
        calculate distance from corner to atom_coordinate
        :param corner: 1d ndarray of floats, shape(3,)
        :param atom_coordinate:  1d ndarray of floats, shape(3,)
        :return: float
        """
        corner_crd = self._get_corner_crd(corner)
        d = (corner_crd - atom_coordinate)**2
        return np.sqrt(d.sum())

    def _containing_cube(self, atom_coordinate):
        if not self._is_in_grid(atom_coordinate):
            return [], 0, 0

        tmp = atom_coordinate - self._origin_crd
        lower_corner = np.array(tmp / spacing, dtype=int)
        eight_corners = [lower_corner + shift for shift in self._eight_corner_shifts]

        distances = []
        for corner in eight_corners:
            distances.append(self._distance(corner, atom_coordinate) )

        nearest_ix = np.argmin(distances)
        furthest_ix = np.argmax(distances)
        return eight_corners, nearest_ix, furthest_ix
    
    def _is_row_in_matrix(self, row, matrix):
        for r in matrix:
            if np.all((row == r)):
                return True
        return False

    def _move_molecule_to_grid_center(self):
        """
        move the molecule to near the grid center
        store self._max_grid_indices
        TODO bugs
        """
        print("Move molecule to grid center.")

        lower_molecule_corner_crd = self._crd.min(axis=0) - 1.5 * self._spacing
        print("Before moving, lower_molecule_corner_crd at", lower_molecule_corner_crd)

        upper_molecule_corner_crd = self._crd.max(axis=0) + 1.5 * self._spacing
        print("Before moving, upper_molecule_corner_crd at", upper_molecule_corner_crd)

        molecule_box_center = (lower_molecule_corner_crd + upper_molecule_corner_crd) / 2.
        grid_center = (self._origin_crd + self._upper_most_corner_crd) / 2.
        displacement = grid_center - molecule_box_center

        print("Molecule is translated by ", displacement)
        self._crd += displacement.reshape(1, 3)

        lower_molecule_corner_crd = self._crd.min(axis=0) - 1.5 * self._spacing
        print("After moving, lower_molecule_corner_crd at", lower_molecule_corner_crd)

        upper_molecule_corner_crd = self._crd.max(axis=0) + 1.5 * self._spacing
        print("After moving, upper_molecule_corner_crd at", upper_molecule_corner_crd)

        molecule_box_lengths = upper_molecule_corner_crd - lower_molecule_corner_crd
        if np.any(molecule_box_lengths < 0):
            raise RuntimeError("One of the molecule box lengths are negative")

        max_grid_indices = np.ceil(molecule_box_lengths / self._spacing)
        print("max_grid_indices", max_grid_indices)

        # self._max_grid_indices is how far it can step before moving out of the box
        self._max_grid_indices = self._grid["counts"] - np.array(max_grid_indices, dtype=int)
        if np.any(self._max_grid_indices <= 1):
            raise RuntimeError("At least one of the max grid indices is <= one")

        return None

    def _move_molecule_to_lower_corner(self):
        """
        move the molecule to near the grid lower corner
        store self._max_grid_indices
        """
        print("Move molecule to lower corner.")

        lower_molecule_corner_crd = self._crd.min(axis=0) - 1.5 * self._spacing
        print("Before moving, lower_molecule_corner_crd at", lower_molecule_corner_crd)

        displacement = self._origin_crd - lower_molecule_corner_crd
        print("Molecule is translated by ", displacement)
        self._crd += displacement.reshape(1, 3)

        lower_molecule_corner_crd = self._crd.min(axis=0) - 1.5 * self._spacing
        print("After moving, lower_molecule_corner_crd at", lower_molecule_corner_crd)
        upper_molecule_corner_crd = self._crd.max(axis=0) + 1.5 * self._spacing
        print("After moving, upper_molecule_corner_crd at", lower_molecule_corner_crd)

        molecule_box_lengths = upper_molecule_corner_crd - lower_molecule_corner_crd
        if np.any(molecule_box_lengths < 0):
            raise RuntimeError("One of the molecule box lengths are negative")

        max_grid_indices = np.ceil(molecule_box_lengths / self._spacing)
        # self._max_grid_indices is how far it can step before moving out of the box
        self._max_grid_indices = self._grid["counts"] - np.array(max_grid_indices, dtype=int)
        if np.any(self._max_grid_indices <= 1):
            raise RuntimeError("At least one of the max grid indices is <= one")

        return None

    def get_grid_func_names(self):
        return self._grid_func_names
    
    def get_grids(self):
        return self._grid
    
    def get_crd(self):
        return self._crd
    
    def get_prmtop(self):
        return self._prmtop
    
    def get_charges(self):
        charges = dict()
        for key in ["CHARGE_E_UNIT", "R_LJ_CHARGE", "A_LJ_CHARGE"]:
            charges[key] = self._prmtop[key]
        return charges

    def get_natoms(self):
        return self._prmtop["POINTERS"]["NATOM"]

    def get_initial_com(self):
        return self._initial_com


def debye_huckel_kappa(I_mole_per_litter, dielectric_constant, temperature):
    """
    :param I_mole_per_litter: float, ionic strength
    :param dielectric_constant: float, dimensionless
    :param temperature: float, temperature in K
    :return: float, kappa (1 / Angstrom)

    Fomula: see https://en.wikipedia.org/wiki/Debye_length, section: In an electrolyte solution
    kappa = sqrt( 2 * NA * e**2 * I / (epsilon_r * epsilon_0 * kB * T)  )
    Where in the SI units:
            NA = NA_bar * 10**(23) [mol**(-1)]; NA_bar = 6.022140857
            e = e_bar * 10**(-19) [C]; e_bar = 1.6021766208
            I: input ionic strength in [mol * m**(-3)]
            epsilon_r: input dielectric constant > 0
            epsilon_0 = epsilon_0_bar * 10**(-12) [C**2 * J**(-2) * m**(-2)], epsilon_0_bar = 8.854187817
            kB = kB_bar * 10**(-23) [J * K**(-1)]
            T: input temperature in K
    """
    assert dielectric_constant > 0, "dielectric_constant must be positive"

    NA_bar = 6.022140857
    e_bar = 1.6021766208
    I_mole_per_m3 = 1000 * I_mole_per_litter
    epsilon_r = dielectric_constant
    epsilon_0_bar = 8.854187817
    kB_bar = 1.38064852
    T = temperature

    kappa = np.sqrt(2 * NA_bar * e_bar**2 * I_mole_per_m3 / (epsilon_r * epsilon_0_bar * kB_bar * T))
    return kappa


class PotentialGrid(Grid):
    """
    calculate the potential part of the interaction energy.
    """
    def __init__(self,
                 prmtop_file_name,
                 inpcrd_file_name,
                 grid_nc_file,
                 new_calculation=False,
                 lj_sigma_scaling_factor=1.,
                 lj_depth_scaling_factor=1.,
                 ionic_strength=0.,
                 dielectric=1.,
                 temperature=300.,
                 spacing=0.25,
                 counts=(1, 1, 1),
                 where_to_place_molecule="center"):
        """
        :param prmtop_file_name: str,  AMBER prmtop file
        :param inpcrd_file_name: str, AMBER coordinate file
        :param grid_nc_file: str, netCDF4 file
        :param new_calculation: if True do the new grid calculation else load data in grid_nc_file
        :param lj_sigma_scaling_factor: float
        :param lj_depth_scaling_factor: float, used to rescale epsilon (LJ depth) to account for different surface tension
        :param ionic_strength: float, in mole per litter
        :param dielectric: float, > 0
        :param temperature: float
        :param spacing: float, in angstrom
        :param counts: tuple of 3 floats, number of grid points a long x, y and z
        :param extra_buffer: float in angstrom,
                            extra buffer space arround the molecule, not used if counts is specified
        :param where_to_place_molecule: str, one of the two ["center", "lower_corner"]
        """
        assert where_to_place_molecule in ["center", "lower_corner"], "Unknown where_to_place_molecule"

        Grid.__init__(self)
        self._FFTs = {}

        # save parameter to self._prmtop
        # save "lj_sigma_scaling_factor" and "lj_depth_scaling_factor" to self._grid
        self._load_prmtop(prmtop_file_name, lj_sigma_scaling_factor, lj_depth_scaling_factor)

        # Charges in potential grid will carry the dielectric
        self._prmtop["CHARGE_E_UNIT"] /= dielectric
        self._set_grid_key_value("dielectric", np.array([dielectric], dtype=float))

        if new_calculation:

            print("Create new netCDF4 file: " + grid_nc_file)
            self._nc_handle = nc.Dataset(grid_nc_file, "w", format="NETCDF4")

            # coordinates stored in self._crd
            self._load_inpcrd(inpcrd_file_name)

            # save "origin", "d0", "d1", "d2", "spacing" and "counts" to self._grid
            self._cal_grid_parameters(spacing, counts)

            # save "x", "y" and "z" to self._grid
            self._cal_grid_coordinates()

            # set convenient parameters for latter use: self._origin_crd, self._upper_most_corner_crd,
            # self._upper_most_corner, self._spacing
            self._initialize_convenient_para()

            # move molecule
            # also stores self._max_grid_indices (not use for potential grid)
            if where_to_place_molecule == "center":
                self._move_molecule_to_grid_center()
            elif where_to_place_molecule == "lower_corner":
                self._move_molecule_to_lower_corner()

            # store initial center of mass
            self._initial_com = self._get_molecule_center_of_mass()
            self._set_grid_key_value("initial_com", self._initial_com)

            # store max_grid_indices  self._max_grid_indices
            self._set_grid_key_value("max_grid_indices", self._max_grid_indices)

            # Debye Huckel kappa
            self._debye_huckel_kappa = self._cal_debye_huckel_kappa(ionic_strength, dielectric, temperature)
            self._set_grid_key_value("ionic_strength", np.array([ionic_strength], dtype=float))
            self._set_grid_key_value("temperature", np.array([temperature], dtype=float))
            self._set_grid_key_value("debye_huckel_kappa", np.array([self._debye_huckel_kappa], dtype=float))

            self._cal_potential_grids()

            # save every thing to nc file
            self._write_to_nc(self._nc_handle, "crd_placed_in_grid", self._crd)
            for key in self._grid:
                self._write_to_nc(self._nc_handle, key, self._grid[key])

            self._nc_handle.close()

        else:
            self._load_precomputed_grids(grid_nc_file)

    def _load_precomputed_grids(self, grid_nc_file):
        """
        :param grid_nc_file: str, netCDF4 file name
        :return: None
        """
        assert os.path.isfile(grid_nc_file), "%s does not exist" %grid_nc_file

        print("Loading precomputed grid in: " + grid_nc_file)
        nc_handle = nc.Dataset(grid_nc_file, "r")

        keys = nc_handle.variables.keys()
        for key in keys:
            if key not in self._grid_func_names:
                self._set_grid_key_value(key, nc_handle.variables[key][:])

        self._initialize_convenient_para()
        self._crd = self._grid["crd_placed_in_grid"]
        self._initial_com = self._grid["initial_com"]
        self._max_grid_indices = self._grid["max_grid_indices"]
        self._debye_huckel_kappa = self._grid["debye_huckel_kappa"]

        natoms = self._prmtop["POINTERS"]["NATOM"]
        if natoms != self._crd.shape[0]:
            raise RuntimeError("Number of atoms is wrong in " + grid_nc_file)

        for key in self._grid_func_names:
            self._set_grid_key_value(key, nc_handle.variables[key][:])
            self._FFTs[key] = self._cal_FFT(key)
            self._set_grid_key_value(key, None)     # to save memory

        nc_handle.close()
        return None

    def _cal_debye_huckel_kappa(self, I_mole_per_litter, dielectric_constant, temperature):
        """
        :param I_mole_per_litter: float, ionic strength
        :param dielectric_constant: float, dimensionless
        :param temperature: float, temperature in K
        :return: float, kappa (1 / Angstrom)

        Formula: see https://en.wikipedia.org/wiki/Debye_length, section: In an electrolyte solution
        kappa = sqrt( 2 * NA * e**2 * I / (epsilon_r * epsilon_0 * kB * T)  )
        Where in the SI units:
                NA = NA_bar * 10**(23) [mol**(-1)]; NA_bar = 6.022140857
                e = e_bar * 10**(-19) [C]; e_bar = 1.6021766208
                I: input ionic strength in [mol * m**(-3)]
                epsilon_r: input dielectric constant > 0
                epsilon_0 = epsilon_0_bar * 10**(-12) [C**2 * J**(-2) * m**(-2)], epsilon_0_bar = 8.854187817
                kB = kB_bar * 10**(-23) [J * K**(-1)]
                T: input temperature in K
        """
        assert dielectric_constant > 0, "dielectric_constant must be positive"

        NA_bar = 6.022140857
        e_bar = 1.6021766208
        I_mole_per_m3 = 1000 * I_mole_per_litter
        epsilon_r = dielectric_constant
        epsilon_0_bar = 8.854187817
        kB_bar = 1.38064852
        T = temperature

        kappa = np.sqrt(2 * NA_bar * e_bar ** 2 * I_mole_per_m3 / (epsilon_r * epsilon_0_bar * kB_bar * T))
        return kappa

    def _cal_FFT(self, name):
        print("Doing FFT for %s" % name)

        if name not in self._grid_func_names:
            raise RuntimeError("%s is not allowed grid name.")

        FFT = np.fft.fftn(self._grid[name])
        return FFT

    def _write_to_nc(self, nc_handle, key, value):
        write_to_nc(nc_handle, key, value)
        return None

    def _cal_grid_parameters(self, spacing, counts):
        """
        :param spacing: float, unit in angstrom, the same in x, y, z directions
        :param bsite_file: str, the file name of "measured_binding_site.py" from AlGDock pipeline
        :param nc_handle: an instance of netCDF4.Dataset()
        :return: None
        """
        assert len(counts) == 3, "counts must have three numbers"
        for count in counts:
            assert count > 0, "count must be positive integer"

        assert spacing > 0, "spacing must be positive"

        self._set_grid_key_value("origin", np.zeros([3], dtype=float))
        
        self._set_grid_key_value("d0", np.array([spacing, 0, 0], dtype=float))
        self._set_grid_key_value("d1", np.array([0, spacing, 0], dtype=float))
        self._set_grid_key_value("d2", np.array([0, 0, spacing], dtype=float))
        self._set_grid_key_value("spacing", np.array([spacing]*3, dtype=float))
        self._set_grid_key_value("counts", np.array(counts, dtype=int))

        return None
    
    def _cal_grid_parameters_without_bsite(self, spacing, extra_buffer, nc_handle):
        """
        use this when making box encompassing the whole receptor
        spacing:    float, unit in angstrom, the same in x, y, z directions
        extra_buffer: float
        """
        assert spacing > 0 and extra_buffer > 0, "spacing and extra_buffer must be positive"
        self._set_grid_key_value("origin", np.zeros( [3], dtype=float))
        
        self._set_grid_key_value("d0", np.array([spacing, 0, 0], dtype=float))
        self._set_grid_key_value("d1", np.array([0, spacing, 0], dtype=float))
        self._set_grid_key_value("d2", np.array([0, 0, spacing], dtype=float))
        self._set_grid_key_value("spacing", np.array([spacing]*3, dtype=float))
        
        lj_radius = np.array(self._prmtop["LJ_SIGMA"]/2., dtype=float)
        dx = (self._crd[:,0] + lj_radius).max() - (self._crd[:,0] - lj_radius).min()
        dy = (self._crd[:,1] + lj_radius).max() - (self._crd[:,1] - lj_radius).min()
        dz = (self._crd[:,2] + lj_radius).max() - (self._crd[:,2] - lj_radius).min()

        print("Receptor enclosing box [%f, %f, %f]"%(dx, dy, dz))
        print("extra_buffer: %f"%extra_buffer)

        length = max([dx, dy, dz]) + 2.0*extra_buffer
        count = np.ceil(length / spacing) + 1
        
        self._set_grid_key_value("counts", np.array([count]*3, dtype=int))
        print("counts ", self._grid["counts"])
        print("Total box size %f" %((count-1)*spacing))

        for key in ["origin", "d0", "d1", "d2", "spacing", "counts"]:
            self._write_to_nc(nc_handle, key, self._grid[key])
        return None
    
    def _cal_grid_coordinates(self):
        """
        calculate grid coordinates (x,y,z) for each corner,
        save 'x', 'y', 'z' to self._grid
        """
        print("Calculating grid coordinates")
        #
        x = np.zeros(self._grid["counts"][0], dtype=float)
        y = np.zeros(self._grid["counts"][1], dtype=float)
        z = np.zeros(self._grid["counts"][2], dtype=float)
        
        for i in range(self._grid["counts"][0]):
            x[i] = self._grid["origin"][0] + i*self._grid["d0"][0]

        for j in range(self._grid["counts"][1]):
            y[j] = self._grid["origin"][1] + j*self._grid["d1"][1]

        for k in range(self._grid["counts"][2]):
            z[k] = self._grid["origin"][2] + k*self._grid["d2"][2]

        self._set_grid_key_value("x", x)
        self._set_grid_key_value("y", y)
        self._set_grid_key_value("z", z)

        return None

    def _get_charges(self, name):
        assert name in self._grid_func_names, "%s is not allowed"%name

        if name == "electrostatic":
            return 332.05221729 * np.array(self._prmtop["CHARGE_E_UNIT"], dtype=float)
        elif name == "LJa":
            return -2.0 * np.array(self._prmtop["A_LJ_CHARGE"], dtype=float)
        elif name == "LJr":
            return np.array(self._prmtop["R_LJ_CHARGE"], dtype=float)
        elif name == "occupancy":
            return np.array([0], dtype=float)
        else:
            raise RuntimeError("%s is unknown"%name)

    def _cal_potential_grids(self):
        """
        :return: None
        """
        for name in self._grid_func_names:
            print("Calculating %s grid"%name)
            charges = self._get_charges(name)

            if name == "electrostatic":
                grid = c_cal_potential_grid_electrostatic(self._crd,
                                                    self._grid["x"], self._grid["y"], self._grid["z"],
                                                    self._origin_crd,
                                                    self._upper_most_corner_crd, self._upper_most_corner,
                                                    self._grid["spacing"], self._grid["counts"], charges,
                                                    self._prmtop["LJ_SIGMA"], self._debye_huckel_kappa)

            elif name == "LJa":
                grid = c_cal_potential_grid_LJa(self._crd,
                                                self._grid["x"], self._grid["y"], self._grid["z"],
                                                self._origin_crd,
                                                self._upper_most_corner_crd, self._upper_most_corner,
                                                self._grid["spacing"], self._grid["counts"], charges,
                                                self._prmtop["LJ_SIGMA"])

            elif name == "LJr":
                grid = c_cal_potential_grid_LJr(self._crd,
                                                self._grid["x"], self._grid["y"], self._grid["z"],
                                                self._origin_crd,
                                                self._upper_most_corner_crd, self._upper_most_corner,
                                                self._grid["spacing"], self._grid["counts"], charges,
                                                self._prmtop["LJ_SIGMA"])

            elif name == "occupancy":
                grid = c_cal_potential_grid_occupancy(self._crd,
                                                self._grid["x"], self._grid["y"], self._grid["z"],
                                                self._origin_crd,
                                                self._upper_most_corner_crd, self._upper_most_corner,
                                                self._grid["spacing"], self._grid["counts"],
                                                self._prmtop["LJ_SIGMA"])

            self._set_grid_key_value(name, grid)
        return None

    def _exact_values(self, coordinate):
        """
        :param coordinate: ndarray of floats, shape = (3,)
        :return: dict
        """
        """
        coordinate: 3-array of float
        calculate the exact "potential" value at any coordinate
        """
        assert len(coordinate) == 3, "coordinate must have len 3"
        if not self._is_in_grid(coordinate):
            raise RuntimeError("atom is outside grid")

        energy_names = [name for name in self._grid_func_names if name != "occupancy"]

        charges = {}
        values = {}
        for name in energy_names:
            charges[name] = self._get_charges(name)
            values[name] = 0.
        
        natoms = self._prmtop["POINTERS"]["NATOM"]
        for atom_ix in range(natoms):
            dif = coordinate - self._crd[atom_ix]
            R = np.sqrt((dif*dif).sum())
            lj_diameter = self._prmtop["LJ_SIGMA"][atom_ix]

            if R > lj_diameter:
                values["electrostatic"] += charges["electrostatic"][atom_ix] / R
                values["LJr"] += charges["LJr"][atom_ix] / R**12
                values["LJa"] += charges["LJa"][atom_ix] / R**6
        
        return values
    
    def direct_energy(self, other_molecule_crd, other_molecule_charges):
        """
        :param other_molecule_crd: ndarray of float with shape (natoms, 3)
        :param other_molecule_charges: ndarray of float with shape (3,)
        :return: float
        """
        assert len(other_molecule_crd) == len(other_molecule_charges["CHARGE_E_UNIT"]), "coord and charges must have the same len"
        energy = 0.
        for atom_ind in range(len(other_molecule_crd)):
            potentials = self._exact_values(other_molecule_crd[atom_ind])
            energy += potentials["electrostatic"] * other_molecule_charges["CHARGE_E_UNIT"][atom_ind]
            energy += potentials["LJr"] * other_molecule_charges["R_LJ_CHARGE"][atom_ind]
            energy += potentials["LJa"] * other_molecule_charges["A_LJ_CHARGE"][atom_ind]
        return energy

    def get_FFTs(self):
        return self._FFTs

    def write_box(self, file_name):
        write_box(self, file_name)
        return None

    def write_pdb(self, file_name, mode):
        write_pdb(self._prmtop, self._crd, file_name, mode)
        return None


class ChargeGrid(Grid):
    """
    Calculate the "charge" part of the interaction energy.
    """

    def __init__(self, prmtop_file_name,
                 inpcrd_file_name,
                 potential_grid,
                 lj_sigma_scaling_factor=1.,
                 lj_depth_scaling_factor=1.,
                 where_to_place_molecule="lower_corner"):
        """
        :param prmtop_file_name: str, name of AMBER prmtop file
        :param inpcrd_file_name: str, name of AMBER coordinate file
        :param potential_grid: an instance of PotentialGrid.
        :param lj_sigma_scaling_factor: float
        :param lj_depth_scaling_factor: float
        :param where_to_place_molecule: str, one of the two ["center", "lower_corner"]
        """
        assert where_to_place_molecule in ["center", "lower_corner"], "Unknown where_to_place_molecule"
        Grid.__init__(self)

        pot_grid_data = potential_grid.get_grids()

        if pot_grid_data["lj_sigma_scaling_factor"][0] != lj_sigma_scaling_factor:
            raise RuntimeError("lj_sigma_scaling_factor is %f, but in potential_grid, it is %f" % (
                lj_sigma_scaling_factor, pot_grid_data["lj_sigma_scaling_factor"][0]))

        if pot_grid_data["lj_depth_scaling_factor"][0] != lj_depth_scaling_factor:
            raise RuntimeError("lj_depth_scaling_factor is %f, but in potential_grid, it is %f" % (
                lj_depth_scaling_factor, pot_grid_data["lj_depth_scaling_factor"][0]))

        exclude_entries = self._grid_func_names + ["initial_com", "max_grid_indices", "crd_placed_in_grid"]
        entries = [key for key in pot_grid_data.keys() if key not in exclude_entries]

        print("Copy entries from receptor_grid \n{}".format(entries))
        for key in entries:
            self._set_grid_key_value(key, pot_grid_data[key])

        self._initialize_convenient_para()

        self._rec_FFTs = potential_grid.get_FFTs()

        self._load_prmtop(prmtop_file_name, lj_sigma_scaling_factor, lj_depth_scaling_factor)
        self._load_inpcrd(inpcrd_file_name)

        # move molecule
        # also stores self._max_grid_indices (not use for potential grid)
        # and self._initial_com
        self._where_to_place_molecule = where_to_place_molecule
        self._move_molecule_to_center_or_corner()

    def _move_molecule_to_center_or_corner(self):
        if self._where_to_place_molecule == "center":
            self._move_molecule_to_grid_center()

        elif self._where_to_place_molecule == "lower_corner":
            self._move_molecule_to_lower_corner()

        self._initial_com = self._get_molecule_center_of_mass()

        return None

    def _get_charges(self, name):
        assert name in self._grid_func_names, "%s is not allowed" % name

        if name == "electrostatic":
            return np.array(self._prmtop["CHARGE_E_UNIT"], dtype=float)
        elif name == "LJa":
            return np.array(self._prmtop["A_LJ_CHARGE"], dtype=float)
        elif name == "LJr":
            return np.array(self._prmtop["R_LJ_CHARGE"], dtype=float)
        elif name == "occupancy":
            return np.array([0], dtype=float)
        else:
            raise RuntimeError("%s is unknown" % name)

    def _cal_charge_grid(self, name):
        charges = self._get_charges(name)
        grid = c_cal_charge_grid(name, self._crd, charges, self._origin_crd,
                                 self._upper_most_corner_crd, self._upper_most_corner,
                                 self._grid["spacing"], self._eight_corner_shifts, self._six_corner_shifts,
                                 self._grid["x"], self._grid["y"], self._grid["z"])
        return grid

    def _cal_corr_func(self, grid_name):
        """
        :param grid_name: str
        :return: fft correlation function
        """
        assert grid_name in self._grid_func_names, "%s is not an allowed grid name" % grid_name

        #grid = self._cal_charge_grid(grid_name)

        #self._set_grid_key_value(grid_name, grid)
        #corr_func = np.fft.fftn(self._grid[grid_name])
        #self._set_grid_key_value(grid_name, None)  # to save memory

        corr_func = np.fft.fftn( self._cal_charge_grid(grid_name) )

        corr_func = corr_func.conjugate()
        corr_func = np.fft.ifftn(self._rec_FFTs[grid_name] * corr_func)
        corr_func = np.real(corr_func)
        return corr_func

    def _cal_energies(self):
        """
        calculate interaction energies
        store self._meaningful_energies which is a dict of 1d-arrays
        meaningful means no boder-crossing and no clashing
        """
        max_i, max_j, max_k = self._max_grid_indices

        #occupancy_corr_func = self._cal_corr_func("occupancy")
        #self._free_of_clash = (occupancy_corr_func < 0.0001)
        #del(occupancy_corr_func) # save memory

        self._free_of_clash = self._cal_corr_func("occupancy")
        self._free_of_clash = (self._free_of_clash < 0.0001)

        # exclude positions where molecule crosses border
        self._free_of_clash = self._free_of_clash[0:max_i, 0:max_j, 0:max_k]

        grid_names = [name for name in self._grid_func_names if name != "occupancy"]
        self._meaningful_energies = {}

        for name in grid_names:
            # exclude positions where ligand crosses border
            self._meaningful_energies[name] = self._cal_corr_func(name)[0:max_i, 0:max_j, 0:max_k]

            # exclude positions where ligand is in clash with receptor, become 1D array
            self._meaningful_energies[name] = self._meaningful_energies[name][self._free_of_clash]

        self._number_of_meaningful_energies = self._meaningful_energies[grid_names[0]].shape[0]

        return None

    def _cal_meaningful_corners(self):
        """
        return grid corners corresponding to self._meaningful_energies
        """
        corners = np.where(self._free_of_clash)
        corners = np.array(corners, dtype=int)
        corners = corners.transpose()
        return corners

    def _place_molecule_crd_in_grid(self, molecular_coord):
        """
        :param molecular_coord: 2d array of floats
        :return: None
        """
        crd = np.array(molecular_coord, dtype=float)
        natoms = self._prmtop["POINTERS"]["NATOM"]

        if (crd.shape[0] != natoms) or (crd.shape[1] != 3):
            raise RuntimeError("Input coord does not have the correct shape.")

        self._crd = crd
        self._move_molecule_to_center_or_corner()
        return None

    def cal_grids(self, molecular_coord=None):
        """
        molecular_coord:    2-array, new liagnd coordinate
        compute charge grids, meaningful_energies, meaningful_corners for molecular_coord
        if molecular_coord==None, self._crd is used
        """
        if molecular_coord is not None:
            self._place_molecule_crd_in_grid(molecular_coord)

        self._cal_energies()
        return None

    def get_number_translations(self):
        return self._max_grid_indices.prod()

    def get_box_volume(self):
        """
        in angstrom ** 3
        """
        spacing = self._grid["spacing"]
        volume = ((self._max_grid_indices - 1) * spacing).prod()
        return volume

    def get_meaningful_energies(self):
        return self._meaningful_energies

    def get_meaningful_corners(self):
        meaningful_corners = self._cal_meaningful_corners()
        if meaningful_corners.shape[0] != self._number_of_meaningful_energies:
            raise RuntimeError("meaningful_corners does not have the same len as self._number_of_meaningful_energies")
        return meaningful_corners

    def set_meaningful_energies_to_none(self):
        self._meaningful_energies = None
        return None

