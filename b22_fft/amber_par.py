

from __future__ import print_function

import os

import numpy as np

class AmberPrmtopLoader(object):
    """
    Load a AMBER prmtop file.
    The format of prmtop files is explained here http://ambermd.org/formats.html.
    All the AMBER prmtop data stored to the dictionary self._parameters which can be accessed by getter methods.
    """
    def __init__(self, prmtop_file_name):
        """
        :param prmtop_file_name: str, file name of the AMBER promtop file.
        """
        assert os.path.exists(prmtop_file_name), "%s does not exist" %prmtop_file_name
        
        self._parameters = dict()
        self._prmtop_file_name = prmtop_file_name
        self._read_flags()
        self._read_parameters()
        self._check_10_12_LJ()
        
        self._conv_pointers_list_2_dict()
        self._set_charges_to_eunit()
        self._set_LJ_grid_charges()
        self._set_resname_and_order()
        self._check_len()
    
    def _read_flags(self):
        """
        Load all the FLAGs from the AMBER prmtop file and store them to self._flags.
        Each FLAG is the name of an AMBER prmtop data field.
        :return: None
        """
        with open(self._prmtop_file_name, "r") as handle:
            self._flags = [ line.split()[-1] for line in handle if "%FLAG" in line ]
        return None
    
    def _read_parameters(self):
        """
        Load paremters for all the FLAGS and store them to the dictionary self._parameters.
        The keys of self._parameters are the same as FLAGS in the prmtop file.
        TODO: this does not look very pythonic
        """
        prmtop_file = open(self._prmtop_file_name, "r")
        records = prmtop_file.read().split("%FLAG ")
        prmtop_file.close()
        max_lengh_of_flags = max([len(flag) for flag in self._flags])

        for flag in self._flags:
            for record in records:
                if flag in record[: max_lengh_of_flags]:
                    #
                    if flag == "ATOM_NAME":
                        tmp = record.replace(flag, '').split("\n")[2: -1]
                        self._parameters[flag] = []
                        for line in tmp:
                            leng = 0
                            while leng < len(line):
                                self._parameters[flag].append(line[leng: leng+4].strip())
                                leng +=4

                    else:
                        tmp = record.replace(flag, "").replace("\n", "").split()
                        data_string = tmp[1:]
                        fortran_format = tmp[0].replace("%FORMAT", "")
                        if "I" in fortran_format:
                            self._parameters[flag] = np.array([int(c) for c in data_string], dtype=int)
                        elif "E" in fortran_format:
                            self._parameters[flag] = np.array([float(c) for c in data_string], dtype=float)
                        else:
                            self._parameters[flag] = data_string
        return None

    def _conv_pointers_list_2_dict(self):
        """
        Reorganize pointers' list into dict
        TODO: avoid hard-coding these keys
        """
        pointers_key = """ NATOM, NTYPES, NBONH,  MBONA,  NTHETH, MTHETA, 
                NPHIH,    MPHIA,  NHPARM, NPARM,  NNB,    NRES, NBONA, 
                NTHETA, NPHIA,  NUMBND, NUMANG, NPTRA, NATYP, 
                NPHB,   IFPERT, NBPER,  NGPER,  NDPER, MBPER,
                MGPER,  MDPER,  IFBOX,  NMXRS, IFCAP, NUMEXTRA """
        pointers_key = pointers_key.replace(",", " ").replace("\n", " ").split()
        
        self._parameters["POINTERS"] = { pointers_key[i] : self._parameters["POINTERS"][i]
                                         for i in range(len(pointers_key)) }
        return None 
    
    def _set_resname_and_order(self):
        """
        Create a new entry "PDB_TEMPLATE" for self._parameters.
        "PDB_TEMPLATE" stores atom names, residue names and residue ordering indices for all atoms.
        """
        names  = []
        order = [] 
        natoms = self._parameters["POINTERS"]["NATOM"]
        res_pointers = self._parameters["RESIDUE_POINTER"]
        res_lables = self._parameters["RESIDUE_LABEL"]
        if len(res_pointers) != len(res_lables):
            raise RuntimeError("len(res_pointers) != len(res_names)")
        # nr_residues = len(res_lables)
        res_pointers = list(res_pointers)
        res_pointers.append(natoms + 1)
        
        for atom_ind in range(natoms):
            i = 0
            found = False
            while not found:
                if ((atom_ind + 1) >= res_pointers[i]) and ((atom_ind + 1) < res_pointers[i + 1]):
                    names.append(res_lables[i])
                    order.append(i + 1)
                    found = True
                i += 1
        
        self._parameters["PDB_TEMPLATE"] = {}
        self._parameters["PDB_TEMPLATE"]["ATOM_NAME"] = self._parameters["ATOM_NAME"]
        self._parameters["PDB_TEMPLATE"]["RES_NAME"] = names
        self._parameters["PDB_TEMPLATE"]["RES_ORDER"] = order
        return None
    
    def _set_charges_to_eunit(self):
        """
        Divide self._parameters['CHARGE'] by 18.2223 and store in self._parameters with the new key "CHARGE_E_UNIT"
        """
        self._parameters["CHARGE_E_UNIT"] = np.array( self._parameters["CHARGE"] ) / 18.2223
        return None
    
    def _set_LJ_grid_charges(self):
        """
        d_ii is diameter; d_ii = ( 2 * A_ii/B_ii )^(1/6) 
        e_ii is depth;    e_ii = ( (B_ii)^2 / (4* A_ii) )
        #
        repulsive_charge = (e_ii)^(1/2) * (d_ii)^6         can be proved that this is   (A_ii)^(1/2)
        atractive_charge = (e_ii)^(1/2) * (d_ii)^3         can be proved that this is   (B_ii / 2)^(1/2)
                                                    when using atractive_charge, multiply by 2 for the potential grid.
        #
        Read this for AMBER's LJ equation and definition of A_ij and B_ij: http://ambermd.org/Questions/vdwequation.pdf
        In amber formulae, r_0 is the most energy-favorable distance between two atoms (LJ_diameter),
        whereas sigma is the distance where the energy is zero.
        Here LJ_diameter is r_0 and LJ_SIGMA is sigma, so LJ_SIGMA = LJ_diameter / (2)^(1/6).
        """
        NTYPES = self._parameters["POINTERS"]["NTYPES"]
        LJ_diameter = np.zeros(NTYPES, dtype=float)
        LJ_depth = np.zeros(NTYPES, dtype=float)
        for i in range(NTYPES):
            LJ_index = self._parameters["NONBONDED_PARM_INDEX"][NTYPES*i+i] - 1
            if self._parameters["LENNARD_JONES_ACOEF"][LJ_index] < 1.0e-6:
                LJ_diameter[i] = 0
                LJ_depth[i] = 0
            else:
                factor = 2. * self._parameters["LENNARD_JONES_ACOEF"][LJ_index] / self._parameters["LENNARD_JONES_BCOEF"][LJ_index]
                LJ_diameter[i] = pow(factor, 1./6)
                LJ_depth[i] = self._parameters["LENNARD_JONES_BCOEF"][LJ_index] / 2. / factor
        
        root_LJ_depth = np.sqrt(LJ_depth)
        
        r_LJ_charge = root_LJ_depth * LJ_diameter**6
        a_LJ_charge = root_LJ_depth * LJ_diameter**3

        # change to arrays with atom index
        natoms = self._parameters["POINTERS"]["NATOM"]
        r_LJ = np.zeros(natoms, dtype=float)
        a_LJ = np.zeros(natoms, dtype=float)
        LJ_sigma = np.zeros(natoms, dtype=float)
        six_root_two = pow(2.0, 1./6)

        for atom_ind in range(natoms):
            type_ind = self._parameters["ATOM_TYPE_INDEX"][atom_ind] - 1
            r_LJ[atom_ind] = r_LJ_charge[type_ind]
            a_LJ[atom_ind] = a_LJ_charge[type_ind]
            LJ_sigma[atom_ind] = LJ_diameter[type_ind] / six_root_two

        # save to dict
        self._parameters["R_LJ_CHARGE"] = r_LJ
        self._parameters["A_LJ_CHARGE"] = a_LJ
        self._parameters["LJ_SIGMA"] = LJ_sigma
        return None
    
    def _check_10_12_LJ(self):
        """
        If any of self._parameters['NONBONDED_PARM_INDEX'] is negative -> 10-12 LJ, throw exception
        """
        if np.any(self._parameters["NONBONDED_PARM_INDEX"] < 0):
            raise RuntimeError("10-12 LJ is used")
        return None
    
    def _check_len(self):
        keys = ["CHARGE_E_UNIT", "R_LJ_CHARGE", "A_LJ_CHARGE", "LJ_SIGMA", "MASS"]
        NATOM = self._parameters["POINTERS"]["NATOM"]
        for key in keys:
            if len(self._parameters[key]) != NATOM:
                raise RuntimeError("%s does not have the same len as NATOM "%key)
        return None

    def get_parm_by_key(self, key):
        return self._parameters[key]
    
    def get_all_parameters(self):
        return self._parameters
    
    def get_parm_for_grid_calculation(self):
        parm = dict()
        keys = ["CHARGE_E_UNIT", "R_LJ_CHARGE", "A_LJ_CHARGE", "LJ_SIGMA", "POINTERS", "MASS", "PDB_TEMPLATE"]
        for key in keys:
            parm[key] = self._parameters[key]
        return parm

    def get_natoms(self):
        return self._parameters["POINTERS"]["NATOM"]


class InpcrdLoader(object):
    """
    Load AMBER or VMD inpcrd coordinate file.
    The coordinates unit is angstrom and stored in self._crd.
    Adapted from alchemicalGrids.py in AlGDock's Pipeline.
    TODO: not to take the box size at the end if it exists.
    """
    def __init__(self, inpcrd_file_name):
        """
        :param inpcrd_file_name: str, name of AMBER coordinate file
        """
        assert os.path.exists(inpcrd_file_name), "%s does not exist" %inpcrd_file_name
        
        if inpcrd_file_name[-3:] == ".gz":
            import gzip
            inpcrd_file = gzip.open(inpcrd_file_name, "r")
        else:
            inpcrd_file = open(inpcrd_file_name, "r")
        inpcrd_lines = inpcrd_file.read().strip().split("\n")
        inpcrd_file.close()

        if "VMD" in inpcrd_lines[0]:
            # VMD format
            self._crd = self._read_vmd_format(inpcrd_lines)
        else:
            self._crd = self._load_amber_format(inpcrd_lines)
        #print("Number of atoms in %s is %d"%(inpcrd_file_name, len(self._crd)))
    
    def _read_vmd_format(self, inpcrd_lines):
        natoms = int(inpcrd_lines[0].split()[-2])

        # remove title
        inpcrd_lines.pop(0)
        crd = []
        for line in inpcrd_lines:
            crd = crd + [float(x) for x in line.split()]
        crd = np.resize(crd, (len(crd)/3, 3))
        if len(crd) > natoms:
            print("box size information included but ignored")
            crd = crd[:natoms, :]
        return crd

    def _load_amber_format(self, inpcrd_lines):
        inpcrd_lines.pop(0)    # remove title
        natoms = int(inpcrd_lines[0])
        inpcrd_lines.pop(0)    # remove number of atoms
        w = 12
        crd = []
        for line in inpcrd_lines:
            crd = crd + [float(line[x: x+w]) for x in range(0, len(line), w)]
        
        crd = np.resize(crd, (len(crd)/3, 3))
        if len( crd ) > natoms:
            print("box size information included but ignored")
            crd = crd[:natoms, :]
        return crd

    def get_coordinates(self):
        return self._crd


