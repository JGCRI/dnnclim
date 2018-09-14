#### Run a linear pattern scaling calculation on a set of input files.
library('assertthat')

run_ps <- function(filelist, varname)
{
    ## The file list should be a vector of string values.  Each string should be
    ## the name of a grid file.  All of the files should be of the same type
    ## (i.e., temperature or precip) The name of the global mean file will be
    ## formed by appending '_gavg.txt' to the name of the grid file.
    
    fld <- do.call(rbind, lapply(filelist, function(fn) {readncfield(fn,varname)}))
    globmean <- do.call(rbind, lapply(filelist, readglobmean))
    
    assert_that(nrow(fld) == nrow(globmean))
    
    ## Separate the data we read into train and test sets
    n <- nrow(fld)
    ntrain = as.integer(round(0.75*n))
    message('ntrain = ', ntrain)
    message('ntest = ', n-ntrain)
    message('grid size = ', ncol(fld))
    
    permvec <- sample(1:n, n)
    trainidx <- sort(permvec[1:ntrain])
    testidx <- sort(permvec[(ntrain+1):n])
    
    ## Fit the linear pattern scaling to the training data
    trainfld <- fld[trainidx,]
    traingmean <- globmean[trainidx,]
    pscl <- fldgen::pscl_analyze(trainfld, traingmean)
    
    ## Apply the pattern scaling coefficients to the test data
    testfld <- fld[testidx,]
    testgmean <- globmean[testidx,]
    psout <- fldgen::pscl_apply(pscl, testgmean)
    
    ## return the mean absolute error
    mean(abs(psout-testfld))
}

readncfield <- function(filename, varname,
                        latvar='lat', lonvar='lon', timevar='time') 
{
    ## This is mostly cribbed from fldgen::read.temperatures
    tann <- ncdf4::nc_open(filename)
    
    ## fld3d should have dimensions of time x lat x lon in the netcdf file.
    ## Because R uses Fortran array layout, this gets reversed to lon x lat x time.
    fld3d <- ncdf4::ncvar_get(tann, var=varname)
    lat <- ncdf4::ncvar_get(tann, var=latvar)
    nlat <- length(lat)
    lon <- ncdf4::ncvar_get(tann, var=lonvar)
    nlon <- length(lon)
    time <- ncdf4::ncvar_get(tann, var=timevar)
    ntime <- length(time)
    timeout <- seq_along(time) - 1
    ncdf4::nc_close(tann)
    
    assert_that(all(dim(fld3d) == c(nlon, nlat, ntime)))

    ## reorganize and flatten the 3-D array into a 2d array of ntime x ngrid
    ## As we do this, we will also make latitude the most rapidly varying index
    ## for the individual time slices.
    
    fld <- aperm(fld3d, c(3,2,1))
    dim(fld) <- c(ntime, nlat*nlon)
    
    ## Return the array.  It has cases in rows and grid cells in columns.
    fld
}


readglobmean <- function(filename, suffix='_gavg.txt')
{
    gfilename <- paste0(filename,suffix)
    tbl <- read.table(gfilename)
    
    ## Return as an nx1 matrix
    matrix(tbl[[1]], ncol=1)
}

