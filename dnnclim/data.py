#! /usr/bin/env python

"""
Functions for importing and exporting data

"""

import netCDF4
import numpy as np
import os.path
import sys

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

def readannualmeans(datain, datadir = '.', fntemplate='{}_Amon_{}_{}_{}_200601-210012.nc_gavg.txt'):
    """
    Read the area-weighted global mean values corresponding to a dataset.

    :param datadir: Directory containing the data
    :param datain: Structure returned by readnetcdf
    :param fntemplate: Template for forming the file name from the varname and scenario
    :return: Nested dictionary of global means corresponding to the data in datain

    The globally averaged data isn't in netCDF files, so we can't rely on the
    netCDF metadata to tell us what models, times, etc, the data correspond to.
    Instead, we use the metadata we collected from the field inputs to decide
    what global averages we should be looking for.  The scenario name is used to
    construct the file name for the global data by substituting into the
    template.  The substitutions will be, in order: 
    (varname, model_id, experiment_id, parent_experiment_rip).
    """

    rslt = {}
    for scenario in datain.keys():
        rslt[scenario] = {} 
        (modelid, exptid, rip) = scenario.split('.')

        for varname in datain[scenario].keys():
            filename = os.path.join(datadir, fntemplate.format(varname, modelid, exptid, rip))

            ## input format is a sequence of values, one for each month.
            f = open(filename,'r')
            vals = np.array([float(x) for x in f.readlines()])
            f.close()

            rslt[scenario][varname] = vals

    return rslt

            
def readtopo(ncfiles):
    """
    Read the land fraction and terrain elevation for the model grid

    :param ncfiles: List of two filenames, one for each variable.  The order
                    doesn't matter.
    :return: Dictionary with entries 'sftlf' and 'orog', corresponding to the
                    two grid variables.

    Unlike the model output, these variables are neither scenario nor time
    specific.  They are model specific, but we don't check whether the model
    matches the model for the output variables.  Failing to provide both
    variables (and only those variables) will cause an exception.
    """

    rslt = {}
    varnames = ['sftlf', 'orog'] # allowed variable names

    for filename in ncfiles:
        ncdata = netCDF4.Dataset(filename)
        filevars = ncdata.variables.keys()
        for vn in varnames:
            if vn in filevars:
                varname = vn
                break
        else:
            ## Didn't find any relevant variable in this file.  Don't report an
            ## error yet.  Maybe it will all work out.
            sys.stderr.write('readtopo:  Warning:  file {} had no relevant variables.  Found variables: {}\n'.format(filename, filevars))
            continue            # can't process this file

        rslt[varname] = np.array(ncdata.variables[varname])

    ## check that we got the right variables
    l = list(rslt.keys())
    l.sort()
    varnames.sort()
    if not(l == varnames):
        sys.stderr.write("readtopo:  Didn't find all required variables.  Expected {}.  Found {}\n".format(varnames, l))
        raise "readtopo: bad input"

    return rslt

    