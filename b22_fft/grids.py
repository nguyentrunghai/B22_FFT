
from __future__ import print_function

import numpy as np
import netcdf4 as nc

from amber_par import AmberPrmtopLoader, InpcrdLoader

from util import c_is_in_grid, cdistance, c_containing_cube
from util import c_cal_charge_grid
from util import c_cal_potential_grid


class Grid(object):
    """
    an base class defining some common methods and data attributes
    working implementations are in LigGrid and RecGrid below
    """
    def __init__(self):
        self._grid = {}

        # TODO add surface
        self._grid_func_names = ("electrostatic", "LJr", "LJa", "occupancy")

        # keys to be stored in the result nc file
        cartesian_axes = ("x", "y", "z")
        box_dim_names = ("d0", "d1", "d2")
        others = ("spacing", "counts", "origin", "lj_sigma_scaling_factor")
        self._grid_allowed_keys = self._grid_func_names + cartesian_axes + box_dim_names + others

        # 8 corners of a cubic box
        self._eight_corner_shifts = [np.array([i,j,k], dtype=int) for i in range(2) for j in range(2) for k in range(2)]
        self._eight_corner_shifts = np.array(self._eight_corner_shifts, dtype=int)

        # 6 corners surrounding a point
        self._six_corner_shifts = self._get_six_corner_shifts()

    def _get_six_corner_shifts(self):
        six_corner_shifts = []
        for i in [-1, 1]:
            six_corner_shifts.append(np.array([i,0,0], dtype=int))
            six_corner_shifts.append(np.array([0,i,0], dtype=int))
            six_corner_shifts.append(np.array([0,0,i], dtype=int))
        return np.array(six_corner_shifts, dtype=int)
    
    def _set_grid_key_value(self, key, value):
        """
        key:    str
        value:  any object
        """
        assert key in self._grid_allowed_keys, key + " is not an allowed key"
        print("setting " + key)
        if key not in self._grid_func_names:
            print(value)
        self._grid[key] = value
        return None
    
    def _load_prmtop(self, prmtop_file_name, lj_sigma_scaling_factor):
        """
        :param prmtop_file_name: str, name of AMBER prmtop file
        :param lj_sigma_scaling_factor: float, must have value in [0.5, 1.0].
        It is stored in self._grid["lj_sigma_scaling_factor"] as
        a array of shape (1,) for reason of saving to nc file.
         Experience says that 0.8 is good for protein-ligand calculations.
        :return: None
        """
        assert 0.5 <= lj_sigma_scaling_factor <= 1.0, "lj_sigma_scaling_factor is out of allowed range"
        self._prmtop = AmberPrmtopLoader(prmtop_file_name).get_parm_for_grid_calculation()

        # scale the Lennard Jones sigma parameters
        self._prmtop["LJ_SIGMA"] *= lj_sigma_scaling_factor

        # save "lj_sigma_scaling_factor" to self._grid
        self._set_grid_key_value("lj_sigma_scaling_factor", np.array([lj_sigma_scaling_factor], dtype=float))
        return None
    
    def _load_inpcrd(self, inpcrd_file_name):
        """
        save the molecule's coordinates to self._crd
        :param inpcrd_file_name: str
        :return: None
        """
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
        store self._max_grid_indices and self._initial_com
        """
        lower_molecule_corner_crd = self._crd.min(axis=0)
        upper_molecule_corner_crd = self._crd.max(axis=0)

        molecule_box_lengths = upper_molecule_corner_crd - lower_molecule_corner_crd
        if np.any(molecule_box_lengths < 0):
            raise RuntimeError("One of the molecule box lengths are negative")

        max_grid_indices = np.ceil(molecule_box_lengths / self._spacing)
        # self._max_grid_indices is how far it can step before moving out of the box
        self._max_grid_indices = self._grid["counts"] - np.array(max_grid_indices, dtype=int)
        if np.any(self._max_grid_indices <= 1):
            raise RuntimeError("At least one of the max grid indices is <= one")

        molecule_box_center = (lower_molecule_corner_crd + upper_molecule_corner_crd) / 2.
        grid_center = (self._origin_crd + self._upper_most_corner_crd) / 2.
        displacement = grid_center - molecule_box_center

        print("Molecule is translated by ", displacement)
        self._crd += displacement.reshape(1, 3)
        self._initial_com = self._get_molecule_center_of_mass()
        return None

    def _move_molecule_to_lower_corner(self):
        """
        move the molecule to near the grid lower corner
        store self._max_grid_indices and self._initial_com
        """
        lower_molecule_corner_crd = self._crd.min(axis=0) - 1.5 * self._spacing
        upper_molecule_corner_crd = self._crd.max(axis=0) + 1.5 * self._spacing

        molecule_box_lengths = upper_molecule_corner_crd - lower_molecule_corner_crd
        if np.any(molecule_box_lengths < 0):
            raise RuntimeError("One of the molecule box lengths are negative")

        max_grid_indices = np.ceil(molecule_box_lengths / self._spacing)
        # self._max_grid_indices is how far it can step before moving out of the box
        self._max_grid_indices = self._grid["counts"] - np.array(max_grid_indices, dtype=int)
        if np.any(self._max_grid_indices <= 1):
            raise RuntimeError("At least one of the max grid indices is <= one")

        displacement = self._origin_crd - lower_molecule_corner_crd
        print("Molecule is translated by ", displacement)
        self._crd += displacement.reshape(1, 3)
        self._initial_com = self._get_molecule_center_of_mass()
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

    def get_allowed_keys(self):
        return self._grid_allowed_keys


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
                 ionic_strength=0.,
                 dielectric=1.,
                 temperature=300.,
                 surface_tension=0.,
                 buried_surface_area_per_atom_pair=0,
                 surface_pair_distance_cutoff=5.,
                 spacing=0.25,
                 counts=(1, 1, 1),
                 where_to_place_molecule="lower_corner"):
        """
        :param prmtop_file_name: str,  AMBER prmtop file
        :param inpcrd_file_name: str, AMBER coordinate file
        :param grid_nc_file: str, netCDF4 file
        :param new_calculation: if True do the new grid calculation else load data in grid_nc_file
        :param lj_sigma_scaling_factor: float 0.5 < lj_sigma_scaling_factor < 1
        :param ionic_strength: float, in mole per litter
        :param dielectric: float, > 0
        :param temperature: float
        :param surface_tension: float, >= 0
        :param buried_surface_area_per_atom_pair: float >=0
        :param surface_pair_distance_cutoff: float in angstrom, > 0.
                                            distance within which a pair is said to form surface contact.
        :param spacing: float, in angstrom
        :param counts: tuple of 3 floats, number of grid points a long x, y and z
        :param extra_buffer: float in angstrom,
                            extra buffer space arround the molecule, not used if counts is specified
        """
        Grid.__init__(self)

        # save parameter to self._prmtop and "lj_sigma_scaling_factor" to self._grid["lj_sigma_scaling_factor"]
        self._load_prmtop(prmtop_file_name, lj_sigma_scaling_factor)
        self._FFTs = {}

        if new_calculation:
            self._load_inpcrd(inpcrd_file_name)
            nc_handle = nc.Dataset(grid_nc_file, "w", format="NETCDF4")

            # TODO calculate grid parameters using spacing and count
            # save "origin", "d0", "d1", "d2", "spacing" and "counts"
            self._cal_grid_parameters(spacing, counts)

            # save "x", "y" and "z" to self._grid
            self._cal_grid_coordinates()

            # set convenient parameters for latter use: self._origin_crd, self._upper_most_corner_crd,
            # self._upper_most_corner, self._spacing
            self._initialize_convenient_para()

            # move molecule and also store self._max_grid_indices (not use for this grid) and self._initial_com
            if where_to_place_molecule == "center":
                self._move_molecule_to_grid_center()
            elif where_to_place_molecule = "lower_corner":
                self._move_molecule_to_grid_lower_corner()

            if bsite_file is not None:
                print("Rececptor is assumed to be correctely translated such that box encloses binding pocket.")
                self._cal_grid_parameters_with_bsite(spacing, bsite_file, nc_handle)
                self._cal_grid_coordinates(nc_handle)
                self._initialize_convenient_para()
            else:
                print("No binding site specified, box encloses the whole receptor")
                self._cal_grid_parameters_without_bsite(spacing, extra_buffer, nc_handle)
                self._cal_grid_coordinates(nc_handle)
                self._initialize_convenient_para()
                self._move_receptor_to_grid_center()

            self._cal_potential_grids(nc_handle)
            self._write_to_nc(nc_handle, "trans_crd", self._crd)
            nc_handle.close()
                
        self._load_precomputed_grids(grid_nc_file, lj_sigma_scaling_factor)

    def _load_precomputed_grids(self, grid_nc_file, lj_sigma_scaling_factor):
        """
        nc_file_name:   str
        lj_sigma_scaling_factor: float, used for consistency check
        load netCDF file, populate self._grid with all the data fields 
        """
        assert os.path.isfile(grid_nc_file), "%s does not exist" %grid_nc_file

        print(grid_nc_file)
        nc_handle = netCDF4.Dataset(grid_nc_file, "r")
        keys = [key for key in self._grid_allowed_keys if key not in self._grid_func_names]
        for key in keys:
            self._set_grid_key_value(key, nc_handle.variables[key][:])

        if self._grid["lj_sigma_scaling_factor"][0] != lj_sigma_scaling_factor:
            raise RuntimeError("lj_sigma_scaling_factor is %f but in %s, it is %f" %(
                lj_sigma_scaling_factor, grid_nc_file, self._grid["lj_sigma_scaling_factor"][0]))

        self._initialize_convenient_para()

        natoms = self._prmtop["POINTERS"]["NATOM"]
        if natoms != nc_handle.variables["trans_crd"].shape[0]:
            raise RuntimeError("Number of atoms is wrong in %s"%nc_file_name)
        self._crd = nc_handle.variables["trans_crd"][:]

        for key in self._grid_func_names:
            self._set_grid_key_value(key, nc_handle.variables[key][:])
            self._FFTs[key] = self._cal_FFT(key)
            self._set_grid_key_value(key, None)     # to save memory

        nc_handle.close()
        return None

    def _cal_FFT(self, name):
        if name not in self._grid_func_names:
            raise RuntimeError("%s is not allowed.")
        print("Doing FFT for %s"%name)
        FFT = np.fft.fftn(self._grid[name])
        return FFT

    def _write_to_nc(self, nc_handle, key, value):
        print("Writing %s into nc file"%key)
        # create dimensions
        for dim in value.shape:
            dim_name = "%d"%dim
            if dim_name not in nc_handle.dimensions.keys():
                nc_handle.createDimension(dim_name, dim)

        # create variable
        if value.dtype == int:
            store_format = "i8"
        elif value.dtype == float:
            store_format = "f8"
        else:
            raise RuntimeError("unsupported dtype %s"%value.dtype)
        dimensions = tuple(["%d"%dim for dim in value.shape])
        nc_handle.createVariable(key, store_format, dimensions)

        # save data
        nc_handle.variables[key][:] = value
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
        print("calculating grid coordinates")
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

    def _cal_potential_grids(self, nc_handle):
        """
        use cython to calculate each to the grids, save them to nc file
        """
        for name in self._grid_func_names:
            print("calculating %s grid"%name)
            charges = self._get_charges(name)
            grid = c_cal_potential_grid(name, self._crd, 
                                        self._grid["x"], self._grid["y"], self._grid["z"],
                                        self._origin_crd, self._uper_most_corner_crd, self._uper_most_corner,
                                        self._grid["spacing"], self._grid["counts"], 
                                        charges, self._prmtop["LJ_SIGMA"])

            self._write_to_nc(nc_handle, name, grid)
            self._set_grid_key_value(name, grid)
            #self._set_grid_key_value(name, None)     # to save memory
        return None
    
    def _exact_values(self, coordinate):
        """
        coordinate: 3-array of float
        calculate the exact "potential" value at any coordinate
        """
        assert len(coordinate) == 3, "coordinate must have len 3"
        if not self._is_in_grid(coordinate):
            raise RuntimeError("atom is outside grid even after pbc translated")
        
        values = {}
        for name in self._grid_func_names:
            if name != "occupancy":
                values[name] = 0.
        
        NATOM = self._prmtop["POINTERS"]["NATOM"]
        for atom_ind in range(NATOM):
            dif = coordinate - self._crd[atom_ind]
            R = np.sqrt((dif*dif).sum())
            lj_diameter = self._prmtop["LJ_SIGMA"][atom_ind]

            if R > lj_diameter:
                values["electrostatic"] +=  332.05221729 * self._prmtop["CHARGE_E_UNIT"][atom_ind] / R
                values["LJr"] +=  self._prmtop["R_LJ_CHARGE"][atom_ind] / R**12
                values["LJa"] += -2. * self._prmtop["A_LJ_CHARGE"][atom_ind] / R**6
        
        return values
    
    def _trilinear_interpolation( self, grid_name, coordinate ):
        """
        grid_name is a str one of "electrostatic", "LJr" and "LJa"
        coordinate is an array of three numbers
        trilinear interpolation
        https://en.wikipedia.org/wiki/Trilinear_interpolation
        """
        raise RuntimeError("Do not use, not tested yet")
        assert len(coordinate) == 3, "coordinate must have len 3"
        
        eight_corners, nearest_ind, furthest_ind = self._containing_cube( coordinate ) # throw exception if coordinate is outside
        lower_corner = eight_corners[0]
        
        (i0, j0, k0) = lower_corner
        (i1, j1, k1) = (i0 + 1, j0 + 1, k0 + 1)
        
        xd = (coordinate[0] - self._grid["x"][i0,j0,k0]) / (self._grid["x"][i1,j1,k1] - self._grid["x"][i0,j0,k0])
        yd = (coordinate[1] - self._grid["y"][i0,j0,k0]) / (self._grid["y"][i1,j1,k1] - self._grid["y"][i0,j0,k0])
        zd = (coordinate[2] - self._grid["z"][i0,j0,k0]) / (self._grid["z"][i1,j1,k1] - self._grid["z"][i0,j0,k0])
        
        c00 = self._grid[grid_name][i0,j0,k0]*(1. - xd) + self._grid[grid_name][i1,j0,k0]*xd
        c10 = self._grid[grid_name][i0,j1,k0]*(1. - xd) + self._grid[grid_name][i1,j1,k0]*xd
        c01 = self._grid[grid_name][i0,j0,k1]*(1. - xd) + self._grid[grid_name][i1,j0,k1]*xd
        c11 = self._grid[grid_name][i0,j1,k1]*(1. - xd) + self._grid[grid_name][i1,j1,k1]*xd
        
        c0 = c00*(1. - yd) + c10*yd
        c1 = c01*(1. - yd) + c11*yd
        
        c = c0*(1. - zd) + c1*zd
        return c
    
    def direct_energy(self, ligand_coordinate, ligand_charges):
        """
        :param ligand_coordinate: ndarray of shape (natoms, 3)
        :param ligand_charges: ndarray of shape (3,)
        :return: dic
        """
        assert len(ligand_coordinate) == len(ligand_charges["CHARGE_E_UNIT"]), "coord and charges must have the same len"
        energy = 0.
        for atom_ind in range(len(ligand_coordinate)):
            potentials = self._exact_values(ligand_coordinate[atom_ind])
            energy += potentials["electrostatic"]*ligand_charges["CHARGE_E_UNIT"][atom_ind]
            energy += potentials["LJr"]*ligand_charges["R_LJ_CHARGE"][atom_ind]
            energy += potentials["LJa"]*ligand_charges["A_LJ_CHARGE"][atom_ind]
        return energy
    
    def interpolated_energy(self, ligand_coordinate, ligand_charges):
        """
        ligand_coordinate:  array of shape (natoms, 3)
        ligand_charges: array of shape (3)
        assume that ligand_coordinate is inside grid
        """
        raise RuntimeError("Do not use, not tested yet")
        assert len(ligand_coordinate) == len(ligand_charges["CHARGE_E_UNIT"]), "coord and charges must have the same len"  
        grid_names = [name for name in self._grid_func_names if name != "occupancy"]
        energy = 0.
        potentials = {}
        for atom_ind in range(len(ligand_coordinate)):
            for name in grid_names:
                potentials[name] = self._trilinear_interpolation(name, ligand_coordinate[atom_ind])
            
            energy += potentials["electrostatic"]*ligand_charges["CHARGE_E_UNIT"][atom_ind]
            energy += potentials["LJr"]*ligand_charges["R_LJ_CHARGE"][atom_ind]
            energy += potentials["LJa"]*ligand_charges["A_LJ_CHARGE"][atom_ind]
        
        return energy

    def get_FFTs(self):
        return self._FFTs

    def write_box(self, file_name):
        IO.write_box(self, file_name)
        return None

    def write_pdb(self, file_name, mode):
        IO.write_pdb(self._prmtop, self._crd, file_name, mode)
        return None


class LigGrid(Grid):
    """
    Calculate the "charge" part of the interaction energy.
    """

    def __init__(self, prmtop_file_name, lj_sigma_scaling_factor,
                 inpcrd_file_name, receptor_grid):
        """
        :param prmtop_file_name: str, name of AMBER prmtop file
        :param lj_sigma_scaling_factor: float
        :param inpcrd_file_name: str, name of AMBER coordinate file
        :param receptor_grid: an instance of RecGrid class.
        """
        Grid.__init__(self)
        grid_data = receptor_grid.get_grids()
        if grid_data["lj_sigma_scaling_factor"][0] != lj_sigma_scaling_factor:
            raise RuntimeError("lj_sigma_scaling_factor is %f but in receptor_grid, it is %f" % (
                lj_sigma_scaling_factor, grid_data["lj_sigma_scaling_factor"][0]))

        entries = [key for key in grid_data.keys() if key not in self._grid_func_names]
        print("Copy entries from receptor_grid", entries)
        for key in entries:
            self._set_grid_key_value(key, grid_data[key])
        self._initialize_convenient_para()

        self._rec_FFTs = receptor_grid.get_FFTs()

        self._load_prmtop(prmtop_file_name, lj_sigma_scaling_factor)
        self._load_inpcrd(inpcrd_file_name)
        self._move_ligand_to_lower_corner()

    def _move_ligand_to_lower_corner(self):
        """
        move ligand to near the grid lower corner
        store self._max_grid_indices and self._initial_com
        """
        spacing = self._grid["spacing"]
        lower_ligand_corner = np.array([self._crd[:, i].min() for i in range(3)], dtype=float) - 1.5 * spacing
        upper_ligand_corner = np.array([self._crd[:, i].max() for i in range(3)], dtype=float) + 1.5 * spacing
        #
        ligand_box_lenghts = upper_ligand_corner - lower_ligand_corner
        if np.any(ligand_box_lenghts < 0):
            raise RuntimeError("One of the ligand box lenghts are negative")

        max_grid_indices = np.ceil(ligand_box_lenghts / spacing)
        self._max_grid_indices = self._grid["counts"] - np.array(max_grid_indices, dtype=int)
        if np.any(self._max_grid_indices <= 1):
            raise RuntimeError("At least one of the max grid indices is <= one")

        displacement = self._origin_crd - lower_ligand_corner
        for atom_ind in range(len(self._crd)):
            self._crd[atom_ind] += displacement

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
                                 self._uper_most_corner_crd, self._uper_most_corner,
                                 self._grid["spacing"], self._eight_corner_shifts, self._six_corner_shifts,
                                 self._grid["x"], self._grid["y"], self._grid["z"])
        return grid

    def _cal_corr_func(self, grid_name):
        """
        :param grid_name: str
        :return: fft correlation function
        """
        assert grid_name in self._grid_func_names, "%s is not an allowed grid name" % grid_name
        grid = self._cal_charge_grid(grid_name)

        self._set_grid_key_value(grid_name, grid)
        corr_func = np.fft.fftn(self._grid[grid_name])
        self._set_grid_key_value(grid_name, None)  # to save memory

        corr_func = corr_func.conjugate()
        corr_func = np.fft.ifftn(self._rec_FFTs[grid_name] * corr_func)
        corr_func = np.real(corr_func)
        return corr_func

    def _do_forward_fft(self, grid_name):
        assert grid_name in self._grid_func_names, "%s is not an allowed grid name" % grid_name
        grid = self._cal_charge_grid(grid_name)
        self._set_grid_key_value(grid_name, grid)
        forward_fft = np.fft.fftn(self._grid[grid_name])
        self._set_grid_key_value(grid_name, None)  # to save memory
        return forward_fft

    def _cal_corr_funcs(self, grid_names):
        """
        :param grid_names: list of str
        :return:
        """
        assert type(grid_names) == list, "grid_names must be a list"

        grid_name = grid_names[0]
        forward_fft = self._do_forward_fft(grid_name)
        corr_func = self._rec_FFTs[grid_name] * forward_fft.conjugate()

        for grid_name in grid_names[1:]:
            forward_fft = self._do_forward_fft(grid_name)
            corr_func += self._rec_FFTs[grid_name] * forward_fft.conjugate()

        corr_func = np.fft.ifftn(corr_func)
        corr_func = np.real(corr_func)
        return corr_func

    def _cal_energies(self):
        """
        calculate interaction energies
        store self._meaningful_energies (1-array) and self._meaningful_corners (2-array)
        meaningful means no boder-crossing and no clashing
        TODO
        """
        max_i, max_j, max_k = self._max_grid_indices

        corr_func = self._cal_corr_func("occupancy")
        self._free_of_clash = (corr_func < 0.001)
        self._free_of_clash = self._free_of_clash[0:max_i, 0:max_j,
                              0:max_k]  # exclude positions where ligand crosses border

        self._meaningful_energies = np.zeros(self._grid["counts"], dtype=float)
        if np.any(self._free_of_clash):
            grid_names = [name for name in self._grid_func_names if name != "occupancy"]
            for name in grid_names:
                self._meaningful_energies += self._cal_corr_func(name)

        self._meaningful_energies = self._meaningful_energies[0:max_i, 0:max_j,
                                    0:max_k]  # exclude positions where ligand crosses border

        self._meaningful_energies = self._meaningful_energies[
            self._free_of_clash]  # exclude positions where ligand is in clash with receptor, become 1D array
        self._number_of_meaningful_energies = self._meaningful_energies.shape[0]

        return None

    def _cal_energies_NOT_USED(self):
        """
        calculate interaction energies
        store self._meaningful_energies (1-array) and self._meaningful_corners (2-array)
        meaningful means no boder-crossing and no clashing
        TODO
        """
        max_i, max_j, max_k = self._max_grid_indices

        corr_func = self._cal_corr_func("occupancy")
        self._free_of_clash = (corr_func < 0.001)
        self._free_of_clash = self._free_of_clash[0:max_i, 0:max_j,
                              0:max_k]  # exclude positions where ligand crosses border

        if np.any(self._free_of_clash):
            grid_names = [name for name in self._grid_func_names if name != "occupancy"]
            self._meaningful_energies = self._cal_corr_funcs(grid_names)
        else:
            self._meaningful_energies = np.zeros(self._grid["counts"], dtype=float)

        self._meaningful_energies = self._meaningful_energies[0:max_i, 0:max_j,
                                    0:max_k]  # exclude positions where ligand crosses border
        self._meaningful_energies = self._meaningful_energies[
            self._free_of_clash]  # exclude positions where ligand is in clash with receptor, become 1D array
        self._number_of_meaningful_energies = self._meaningful_energies.shape[0]
        return None

    def _cal_meaningful_corners(self):
        """
        return grid corners corresponding to self._meaningful_energies
        """
        corners = np.where(self._free_of_clash)
        corners = np.array(corners, dtype=int)
        corners = corners.transpose()
        return corners

    def _place_ligand_crd_in_grid(self, molecular_coord):
        """
        molecular_coord:    2-array, new liagnd coordinate
        """
        crd = np.array(molecular_coord, dtype=float)
        natoms = self._prmtop["POINTERS"]["NATOM"]
        if (crd.shape[0] != natoms) or (crd.shape[1] != 3):
            raise RuntimeError("Input coord does not have the correct shape.")
        self._crd = crd
        self._move_ligand_to_lower_corner()
        return None

    def cal_grids(self, molecular_coord=None):
        """
        molecular_coord:    2-array, new liagnd coordinate
        compute charge grids, meaningful_energies, meaningful_corners for molecular_coord
        if molecular_coord==None, self._crd is used
        """
        if molecular_coord is not None:
            self._place_ligand_crd_in_grid(molecular_coord)
        else:
            self._move_ligand_to_lower_corner()  # this is just in case the self._crd is not at the right position

        self._cal_energies()
        return None

    def get_bpmf(self, kB=0.001987204134799235, temperature=300.0):
        """
        use self._meaningful_energies to calculate and return exponential mean
        """
        if len(self._meaningful_energies) == 0:
            return 0.

        beta = 1. / temperature / kB
        V_0 = 1661.

        nr_samples = self.get_number_translations()
        energies = -beta * self._meaningful_energies
        e_max = energies.max()
        exp_mean = np.exp(energies - e_max).sum() / nr_samples

        bpmf = -temperature * kB * (np.log(exp_mean) + e_max)

        V_binding = self.get_box_volume()
        correction = -temperature * kB * np.log(V_binding / V_0 / 8 / np.pi ** 2)
        return bpmf + correction

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

    def get_initial_com(self):
        return self._initial_com


if __name__ == "__main__":
    # do some test
    rec_prmtop_file = "../examples/amber/t4_lysozyme/receptor_579.prmtop"
    rec_inpcrd_file = "../examples/amber/t4_lysozyme/receptor_579.inpcrd"
    grid_nc_file = "../examples/grid/t4_lysozyme/grid.nc"
    lj_sigma_scaling_factor = 0.8
    bsite_file = "../examples/amber/t4_lysozyme/measured_binding_site.py"
    spacing = 0.25

    rec_grid = RecGrid(rec_prmtop_file, lj_sigma_scaling_factor, rec_inpcrd_file, 
                        bsite_file,
                        grid_nc_file,
                        new_calculation=True,
                        spacing=spacing)
    print("get_grid_func_names", rec_grid.get_grid_func_names())
    print("get_grids", rec_grid.get_grids())
    print("get_crd", rec_grid.get_crd())
    print("get_prmtop", rec_grid.get_prmtop())
    print("get_prmtop", rec_grid.get_charges())
    print("get_natoms", rec_grid.get_natoms())
    print("get_natoms", rec_grid.get_allowed_keys())

    rec_grid.write_box("../examples/grid/t4_lysozyme/box.pdb")
    rec_grid.write_pdb("../examples/grid/t4_lysozyme/test.pdb", "w")

    lig_prmtop_file = "../examples/amber/benzene/ligand.prmtop"
    lig_inpcrd_file = "../examples/amber/benzene/ligand.inpcrd"
    lig_grid = LigGrid(lig_prmtop_file, lj_sigma_scaling_factor, lig_inpcrd_file, rec_grid)
    lig_grid.cal_grids()
    print("get_bpmf", lig_grid.get_bpmf())
    print("get_number_translations", lig_grid.get_number_translations())
    print("get_box_volume", lig_grid.get_box_volume())
    print("get_meaningful_energies", lig_grid.get_meaningful_energies())
    print("get_meaningful_corners", lig_grid.get_meaningful_corners())
    print("set_meaningful_energies_to_none", lig_grid.set_meaningful_energies_to_none())
    print("get_initial_com", lig_grid.get_initial_com())

