#! /usr/bin/env python

"""
Functions for importing and exporting data

"""

import netCDF4
import numpy as np
import os.path
import sys
import pickle

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

def readglobmeans(datain, datadir = '.', fntemplate='{}_Amon_{}_{}_{}_200601-210012.nc_gavg.txt'):
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
    if not (l == varnames):
        sys.stderr.write("readtopo:  Didn't find all required variables.  Expected {}.  Found {}\n".format(varnames, l))
        raise "readtopo: bad input"

    return rslt

    
def chkdata(esmvars, globmeans, topo, monthly=True):
    """
    Check that the data read in from the other functions is compatible.

    :param esmvars: dictionary of ESM outputs returned by readncfiles
    :param globmeans: dictionary of globally averaged ESM variables returned by readglobmeans
    :param topo: dictionary of topographic data returned by readtopo
    :param monthly: flag indicating that the data is monthly

    We check the data for the following consistency properties:
       1. All ESM fields and global means have the same time dimension.
       2. Each ESM field has a corresponding global mean time series.
       3. Each temperature field has a corresponding precip field and vice versa.
       4. All ESM fields and topo fields have the same spatial dimensions.

    If the monthly flag is set, we additionally check that:
       5. The number of time steps is a multiple of 12.

    """

    scens = list(esmvars.keys())
    (nt, nlat, nlon) = esmvars[scens[0]]['tas'].shape

    ## property 5, if applicable
    if monthly and nt % 12 != 0:
        raise "chkdata: time series length is not a multiple of 12."
    
    for scen in scens:
        scendata = esmvars[scen]
        ## property 3
        if 'tas' not in scendata or 'pr' not in scendata:
            raise "chkdata: temperature and precipitation not provided for all scenarios."

        ## property 1
        for v in scendata:
            if scendata[v].shape != (nt, nlat, nlon): # also covers property 4 for the ESM fields
                raise "chkdata: dimension mismatch in ESM output."
            if globmeans[scen][v].shape[0] != nt:     # implicitly tests property 2, will raise KeyError if not satisfied
                raise "chkdata: time dimension mismatch in global means."

        ## property 4 for topo fields (ESM fields were handled above)
        for t in topo:
            if topo[t].shape != (nlat, nlon):
                raise "chkdata: dimension mismatch for topo"

    sys.stdout.write('Data checks passed.\n')
    return None


def annualavg(esmvars, globmeans):
    """Compute annual means from monthly data.

    :param esmvars: ESM output fields from readncfiles
    :param globmeans: Global means from readglobmeans
    :return: tuple (esmfld, gmean) of annually averaged fields and global means

    The structures returned will be indexed by scenario and variable, just as
    the input structures were.
    """

    scens = list(esmvars.keys())
    varnames = list(esmvars[scens[0]].keys())

    (nt, nlat, nlon) = esmvars[scens[0]][varnames[0]].shape
    nyear = int(nt / 12)
    dims = (nyear, nlat, nlon)  # output dimensions
    
    
    esmfld = {}
    gmean = {}

    months = np.array(range(12)) # month indexes

    for scen in scens:
        esmfld[scen] = {}
        gmean[scen] = {}
        
        for var in varnames:
            infld = esmvars[scen][var]
            inmean = globmeans[scen][var]
            outfld = np.zeros(dims)
            outmean = np.zeros(nyear)

            for year in range(nyear):
                idx = 12*year + months

                outfld[year,] = np.mean(infld[idx,], axis=0)
                outmean[year,] = np.mean(inmean[idx,], axis=0)

            esmfld[scen][var] = outfld
            gmean[scen][var] = outmean

    return (esmfld, gmean)


    
def preparedata(esmvars, globmeans, topo, trainfrac=0.5, devfrac=0.25):
    """Separate data into test, training, and dev sets, and organize into a single structure.

    :param esmvars: dictionary of ESM outputs returned by readncfiles
    :param globmeans: dictionary of globally averaged ESM variables returned by readglobmeans
    :param topo: dictionary of topographic data returned by readtopo
    :param trainfrac: fraction of the data to include in the training set
    :param devfrac: fraction of the data to include in the dev set

    This function concatenates all of the input fields into a single
    3-d array and all of the global means into a single 1-d array (in
    corresponding order).  These data are randomly split into
    training, dev, and test sets.  All of this is compiled into a
    single structure that is indexed by set (train/dev/test) and data
    type (fld/gmean).  For both the fields and the means, the two
    variables (temperature and precipitation) are stacked into a
    single array.  For example, rslt['dev']['gmean'][:,0] gives the
    global mean temperature values for the dev set,
    rslt['train']['fld'][:,1] gives the precipitation field for the
    training set.  (Note that we store global mean precip even though
    we don't currently use it for anything in the model.)

    The topo data is constant across all cases; therefore, it is not
    split into train/dev/test sets.  The two 2-d topo fields are
    stacked into a single two-channel field (landfrac, elevation) and
    stored as rslt['topo'].

    """

    ## collect a list of ESM fields and global mean values to concatenate
    fields = []
    gmeans = []
    
    for scen in esmvars: 
        field = np.stack([esmvars[scen]['tas'], esmvars[scen]['pr']], axis=1) # result is nt x 2 x nlat x nlon
        gmean = np.stack([globmeans[scen]['tas'], globmeans[scen]['pr']], axis=1) # result is nt x 2

        fields.append(field)
        gmeans.append(gmean)

    ## concatenate all of the stacked fields.  Then transpose them to
    ## put channels in the last dimension, conforming to tensorflow
    ## convention.
    allfields = np.transpose(np.concatenate(fields),
                             axes=[0,2,3,1]) # result is (nscen*nt) x nlat x nlon x 2
    allgmeans = np.concatenate(gmeans)       # result is (nscen*nt) x 2

    ## Conventionally, tensorflow puts channels in the last dimension.
    ## Transpose the arrays to conform.
    

    nscen = len(esmvars)
    dim = esmvars[next(iter(esmvars))]['tas'].shape

    sys.stdout.write('dim fields = {}\n'.format(dim))
    sys.stdout.write('nscen = {}\n'.format(nscen))
    sys.stdout.write('dim allfields = {}\n'.format(allfields.shape))
    sys.stdout.write('dim allgmeans = {}\n'.format(allgmeans.shape))

    ## split in to train/dev/test

    testfrac = 1.0 - (trainfrac+devfrac)
    if not(trainfrac > 0 and trainfrac < 1 and
           devfrac > 0 and devfrac < 1 and
           testfrac > 0 and testfrac < 1):
        raise 'trainfrac, devfrac, and testfrac must all be between 0 and 1'
    
    ncase = allfields.shape[0]
    itrain = int(np.round(trainfrac*ncase))
    idev = itrain + int(np.round(devfrac*ncase))
    
    idx = np.random.permutation(ncase)

    idxtrain = idx[np.arange(itrain)]
    idxdev = idx[np.arange(itrain, idev)]
    idxtest = idx[np.arange(idev, ncase)]

    sys.stdout.write('total cases: {}\n'.format(ncase))
    sys.stdout.write('training cases: {}\n'.format(itrain))
    sys.stdout.write('  {}\n'.format(idxtrain))
    sys.stdout.write('dev cases: {}\n'.format(idev-itrain))
    sys.stdout.write('  {}\n'.format(idxdev))
    sys.stdout.write('test cases: {}\n'.format(ncase-idev))
    sys.stdout.write('  {}\n'.format(idxtest))

    rslt = {}
    for s in ['train','dev','test']:
        rslt[s] = {}

    rslt['train']['fld'] = allfields[idxtrain,]
    rslt['train']['gmean'] = allgmeans[idxtrain,]
    rslt['dev']['fld'] = allfields[idxdev,]
    rslt['dev']['gmean'] = allgmeans[idxdev,]
    rslt['test']['fld'] = allfields[idxtest,]
    rslt['test']['gmean'] = allgmeans[idxtest,]

    ## add in the topo fields.  Stack them with the two channels in the last dimension
    rslt['topo'] = np.stack([topo['sftlf'], topo['orog']], axis=-1)
    

    sys.stdout.write('topo dim:  {}\n'.format(rslt['topo'].shape))
    
    return rslt

def process_dir(inputdir,
                outfile = 'dnnclim.dat',
                tempglob = 'tas*.nc',
                prglob = 'pr*.nc',
                lfglob = 'sftlf*.nc',
                elevglob = 'orog*.nc',
                seed = 8675309):
    """Process and save all of the data in an input directory.

    :inputdir: Directory containing the input data
    :outfile: Name of the file to save the results in. This will go in the current directory, by default
    :tempglob: Glob pattern for the temperature files
    :prglob: Glob pattern for the precipitation files
    :lfglob: Glob pattern for the land fraction file. There should be only one file matching this pattern in the input directory.
    :elevglob: Glob pattern for the elevation file.  There should be only one file matching this pattern in the input directory.
    :seed: Seed for the random number generator used to separate data into train/dev/test sets
    :return: Processed dataset with train/dev/test subsets (see preparedata()).

    This function runs all of the steps defined elsewhere in this
    module, culminating with preparedata().  The resulting data
    structure is pickled and written to the file specified by outfile;
    it is also returned from the function.  The file output can be
    suppressed by passing None as the outfile parameter.

    """
    import glob
    
    def getfilenames(g, unique=False):
        fg = os.path.join(inputdir, g)
        files = glob.glob(fg)
        if len(files) == 0:
            raise 'No files matching {} found.'.format(g)
        if unique and len(files) > 1:
            sys.stderr.write('WARNING: multiple files matching {} found. Using just the first one'.format(g))
            files = files[0:1]
        return files
    
    tempfiles = getfilenames(tempglob)
    prfiles = getfilenames(prglob)
    topofiles = getfilenames(lfglob)
    topofiles += getfilenames(elevglob)

    print('tempfiles : ', tempfiles)
    print('prfiles : ', prfiles)
    print('topofiles : ', topofiles)

    flds = readncfiles(tempfiles, 'tas')
    flds = readncfiles(prfiles, 'pr', datain=flds) # add precip data to previous

    gmeans = readglobmeans(flds, inputdir)

    topo = readtopo(topofiles)

    chkdata(flds, gmeans, topo)

    (flds, gmeans) = annualavg(flds, gmeans)

    rslt = preparedata(flds, gmeans, topo)

    
    if outfile is not None:
        outfile = open(outfile, 'wb') 
        pickle.dump(rslt, outfile)
        outfile.close()

    return rslt
