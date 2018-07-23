"""
define function to read, write pdb files
"""

from amber_par import AmberPrmtopLoader


def write_pdb(prmtop, xyz, pdb_file_name, mode):
    """
    :param prmtop: str or dic returned by AmberPrmtopLoader.get_parm_for_grid_calculation()
    :param xyz: ndarray, the molecular coordinate
    :param pdb_file_name: str
    :param mode: str, either "w" or "a"
    :return: None
    """
    assert mode in ["w", "a"], "unsupported mode"
    out_pdb = open(pdb_file_name, mode)

    if type(prmtop) == str:
        prmtop = AmberPrmtopLoader(prmtop).get_parm_for_grid_calculation()

    natoms = prmtop["POINTERS"]["NATOM"]
    if len(xyz) != natoms:
        raise RuntimeError("Nr atoms in prmtop is %d but in xyz is %d" % (natoms, len(xyz)))
    pdb_template = prmtop["PDB_TEMPLATE"]

    out_pdb.write("MODEL\n")
    for i in range(natoms):
        entry = ("ATOM", (i + 1), pdb_template["ATOM_NAME"][i], pdb_template["RES_NAME"][i],
                 pdb_template["RES_ORDER"][i], xyz[i][0], xyz[i][1], xyz[i][2], 1., 0.)
        out_pdb.write("%4s%7d %4s %3s%6d    %8.3f%8.3f%8.3f%6.2f%6.2f\n" % entry)

    out_pdb.write("TER\nENDMDL\n")
    out_pdb.close()
    return None


def write_box(grid, pdb_file_name):
    """
    :param grid: an object of GridCal
    :param pdb_file_name: str
    :return: None
    """
    out_pdb = open(pdb_file_name, "w")
    grid_data = grid.get_grids()

    origin_crd = grid_data["origin"]
    uper_corner = tuple(grid_data["counts"] - 1)
    i, j, k = uper_corner
    uper_corner_crd = np.array([grid_data["x"][i], grid_data["y"][j], grid_data["z"][k]], dtype=float)
    #
    x = [origin_crd[0], uper_corner_crd[0]]
    y = [origin_crd[1], uper_corner_crd[1]]
    z = [origin_crd[2], uper_corner_crd[2]]
    xyz = [[i, j, k] for i in x for j in y for k in z]
    for i in range(len(xyz)):
        entry = tuple(["ATOM", (i + 1), "DU", (i + 1), "BOX", 1, xyz[i][0], xyz[i][1], xyz[i][2]])
        out_pdb.write("%4s%7d  %2s%d %3s%6d    %8.3f%8.3f%8.3f\n" % entry)

    out_pdb.write("CONECT    1    2    3    5\n")
    out_pdb.write("CONECT    2    1    4    6\n")
    out_pdb.write("CONECT    3    1    4    7\n")
    out_pdb.write("CONECT    4    2    3    8\n")
    out_pdb.write("CONECT    5    1    6    7\n")
    out_pdb.write("CONECT    6    2    5    8\n")
    out_pdb.write("CONECT    7    3    5    8\n")
    out_pdb.write("CONECT    8    4    6    7\n")
    out_pdb.close()
    return None
