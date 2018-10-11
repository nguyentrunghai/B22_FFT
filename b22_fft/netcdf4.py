"""
define functions to read/write netCDF4 files
"""

import numpy as np
import netCDF4 as nc


def load_nc(nc_file_name):
    """
    nc_file_name is a string
    return a dictionary
    """
    assert os.path.isfile(nc_file_name), "%s does not exist" %nc_file_name
    in_nc = nc.Dataset(nc_file_name, "r")
    data = dict()
    for key in in_nc.variables.keys():
        data[key] = in_nc.variables[key][:]
    in_nc.close()
    return data


def write_nc(data, nc_file_name, exclude=()):
    """
    :param data: dic mapping str to ndarray
    :param nc_file_name: str
    :param exclude: tuple or list, which data keys to exclude
    :return: nc handle
    """
    keys = [key for key in data.keys() if key not in exclude]
    out_nc = nc.Dataset(nc_file_name, "w", format="NETCDF4")

    # create dimensions
    for key in keys:
        for dim in data[key].shape:
            dim_name = "%d"%dim
            if dim_name not in out_nc.dimensions.keys():
                out_nc.createDimension( dim_name, dim)

    # create variables
    for key in keys:
        if data[key].dtype == int:
            store_format = "i8"
        elif data[key].dtype == float:
            store_format = "f8"
        else:
            raise RuntimeError("unsupported dtype %s"%data[key].dtype)
        dimensions = tuple([ "%d"%dim for dim in data[key].shape ])
        out_nc.createVariable(key, store_format, dimensions)

    # save data
    for key in keys:
        out_nc.variables[key][:] = data[key]
    return out_nc


def write_to_nc(nc_handle, key, value):
    print("Writing %s into nc file"%key)
    # create dimensions
    for dim in value.shape:
        dim_name = "%d"%dim
        if dim_name not in nc_handle.dimensions.keys():
            nc_handle.createDimension(dim_name, dim)

    # create variable
    if value.dtype == np.int or value.dtype == np.uint16:
        #store_format = "i8"
        # use this because char_trans_corners store a lot of small nonnegative numbers
        # u2 is 16-bit unsigned integer : http://unidata.github.io/netcdf4-python/#section4
        store_format = "u2"
    elif value.dtype == float:
        store_format = "f8"
    else:
        raise RuntimeError("unsupported dtype %s"%value.dtype + " for " + key)
    dimensions = tuple(["%d"%dim for dim in value.shape])
    nc_handle.createVariable(key, store_format, dimensions)

    # save data
    nc_handle.variables[key][:] = value
    return None