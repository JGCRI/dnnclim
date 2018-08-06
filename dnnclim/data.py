#! /usr/bin/env python

"""
Functions for importing and exporting data

"""

import netCDF4
import numpy as np

def readncfiles(ncfiles, varname, fldname=None, datain=None,
                latdim='lat', londim='lon', timedim='time'):
    """Read data from a list of netCDF files.
    
    :param ncfiles: (list of strings) NetCDF files to read
    :param varname: (string) Variable being read in, such as 'tas' for surface temperature.
    :param fldname: (string) Name of the netCDF field that holds the data. (Default: use
                      the value of dtype)
    :param datain: (dictionary) Data from a previous call to this function. The new data
                      will be added to this structure. 
    :param latdim: (string) name of the latitude dimension in the input files (default: 'lat')
    :param londim: (string) name of the longitude dimension in the input files (default: 'lon')
    :param timedim: (string) name of the time dimension in the input files (default: 'time')
    :return: Dictionary of data read from the input files.  See below for the format of 
                the structure.

    This function will read a single field from each file in the input file
    list.  The arrays will be placed into a nested dictionary indexed by
    scenario and variable name.  The scenario name is constructed by
    concatenating the 'model_id', 'experiment_id', and 'parent_experiment_rip'
    metadata values.

    For example, rslt['CESM1-CAM5.rcp60.r1i1p1']['tas'] would hold a 3D array
    containing the surface air temperature for the r1i1p1 run of the RCP 6.0
    scenario of CESM1-CAM5 (a widely-used earth system model).  This indexing
    scheme allows us to match up the data from different files (i.e., for
    different variables) by scenario.

    We are assuming here that all of the fields we read in have the same
    coordinates.  Since we don't store the coordinate variables, there is no way
    of checking this; however, if all of the input files come from the same ESM,
    this assumption should be valid.

    """

    if datain is None:
        rslt = {}
    else:
        rslt = datain

    if fldname is None:
        fldname = varname

    for filename in ncfiles:
        ncdata = netCDF4.Dataset(filename)

        ## Get the metadata to construct the scenario id
        meta = [ncdata.model_id, ncdata.experiment_id, ncdata.parent_experiment_rip]
        scenid = '.'.join(meta)
        if scenid not in rslt:
            rslt[scenid] = {}

        ## Get the data.  The coordinates are [time, lat, lon].
        data = np.array(ncdata.variables[fldname])

        ## Store the entire 3d array under the appropriate scenario and varname
        ## indices.  We are assuming that the coordinates are all the same
        ## across all input files, so we don't have to store the coordinate
        ## variables.  This should be fine as long as all of the data come from
        ## the same ESM.
        rslt[scenid][varname] = data
        
    return rslt
