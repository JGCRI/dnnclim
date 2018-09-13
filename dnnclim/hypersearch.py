"""Functions for searching the space of hyperparameters for an ideal configuration"""

import numpy as np
import copy
import pickle
import dnnclim
import sys

def conjugate(modelspec1, modelspec2, nmutate=1):
    """Produce a hybrid model based on each of two parent models"""
    modlist = [modelspec1, modelspec2]
    choice = np.random.randint(2, size=4)

    newmodel = []
    for i in range(4):
        ## add the ith item from model choice[i] (where choice[i] is in [0,1])
        newmodel.append(modlist[choice[i]][i])

    for i in range(nmutate):
        newmodel = mutate_config(newmodel)

    return tuple(newmodel)


def mutate_config(modelspec):
    """Apply a single mutation to a configuration."""

    modelspec = list(modelspec)
    ibranch = np.random.randint(len(modelspec)) # len(modelspec) should be 4.

    if ibranch < 3:
        modelspec[ibranch] = mutate_branch(modelspec[ibranch])
    else:
        modelspec[ibranch] = mutate_otherparams(modelspec[ibranch])

    return tuple(modelspec)


def mutate_branch(branch):
    """Apply a single mutation to a branch.

    Select a stage at random from the branch and apply a mutation to
    that branch.

    """

    branch = list(branch)
    istage = np.random.randint(len(branch))
    branch[istage] = mutate_stage(branch[istage])

    return tuple(branch)
    

def mutate_stage(stage):
    """Mutate a stage.
    :param stage: structure of the stage to mutate 
    :return: new stage config

    A downsampling (upsampling) stage can be mutated by any one of:
    0. Adding a convolution layer before (after) an existing layer
    1. Deleting one of the convolutional layers
    2. Changing the number of filters in a convolutional layer
    3. Changing the kernel size of a convolutional layer

    """

    stage = list(stage)         # make stage mutable

    ## Decide what kind of mutation we are going to apply.
    if stage[0][0] == 'D':
        ## stage has nothing in it but a single max pool layer; only
        ## option in this case is to add a conv layer
        choice = 0
    else:
        choicevalid = False
        while not choicevalid:
            choice = np.random.randint(4)
            if choice == 1 and len(stage) == 1:
                ## We're trying to delete the only layer in the stage.
                ## That's no good.
                continue
            else:
                choicevalid = True



        
    if choice==0:
        ## Adding a layer
        inew = np.random.randint(len(stage)) # [0, N-1]
        if stage[0][0] == 'U':
            ## in upsampling branches the new layer can't go before
            ## the first one, but it can go after the last one.
            inew += 1
        elif stage[0][0] == 'C' or stage[0][0] == 'D':
            ## This situation is ok; no action required
            pass
        else:
            ## Something wrong with the network structure
            raise ValueError('Invalid stage: {}'.format(stage)) 
            
        stage.insert(inew, newconv())
    else:
        layervalid = False      # indicate whether we have chosen a
                                # valid layer to mutate
        while not layervalid:
            iconv = np.random.randint(len(stage)) 
            if stage[iconv][0] == 'D':
                ## Can't change the max pooling layer in a
                ## downsampling branch; pick again
                continue
            elif stage[iconv][0] == 'U' or stage[iconv][0] == 'C':
                ## This is ok, no action required
                pass
            else:
                ## something wrong.
                raise ValueError('Invalid stage: {}'.format(stage)) 

            ## At this point we are guaranteed that the stage is
            ## either a transpose convolution ('U') or a convolution
            ## ('C')
            
            if choice==1:
                if stage[iconv][0] == 'U':
                    ## Can't delete a transpose convolution layer.
                    ## pick again
                    continue
                else:
                    stage.pop(iconv)               # doink!

            elif choice==2:
                ## Change number of filters in the layer.  This works
                ## the same for convolution and transpose convolution.
                stage[iconv] = mutate_nfilt(stage[iconv])

            elif choice==3:
                ## Change the kernel size.  This is also the same for
                ## the two layer types.
                stage[iconv] = mutate_nkern(stage[iconv])

            ## If we got to here, we selected a valid layer
            layervalid = True 

    return tuple(stage)


def newconv():
    nfilt = np.random.randint(8,17)
    kxsize = np.random.randint(3,8)
    kysize = np.random.randint(3,8)

    return ('C', nfilt, (kxsize, kysize))

def mutate_nfilt(layer):
    chg = 0
    oldn = layer[1]
    while chg==0:
        ## roll 2d6 - 7.  Reroll zeros
        chg = np.sum(np.random.randint(1,7,2)) - 7
        
    newn = np.maximum(oldn + chg, 2) # minimum of 2 filters

    layerout = list(layer)
    layerout[1] = int(newn)

    return tuple(layerout)

def mutate_nkern(layer):
    chgx = 0
    chgy = 0

    oldx = layer[2][0]
    oldy = layer[2][1]

    newx = oldx
    newy = oldy
    
    while newx == oldx and newy == oldy:
        ## roll 2d4 - 5.  Reroll if the result is the same as what we
        ## started with.
        chgx = np.sum(np.random.randint(1,5,2)) - 5
        chgx = np.sum(np.random.randint(1,5,2)) - 5

        ## minimum kernel size is 1
        newx = np.maximum(oldx + chgx, 1)
        newy = np.maximum(oldy + chgy, 1)

    layerout = list(layer)
    layerout[2] = (int(newx), int(newy))
        
    return tuple(layerout)


def mutate_otherparams(op):
    """Mutate the other parameters in a configuration.

    Currently the other parameters include:
    * learning rate (float)
    * regularization (type and float)
    * temperature loss weight (float)
    * precip loss weight (float)

    All of these numerical values are strictly positive, so we mutate
    them by drawing a random normal and adding it to the log of the
    value.  For regularization, 90% of the time we follow this same
    procedure to change the weight of the regularization term.  The
    rest of the time we swap the type of regularization.  
    
    """

    op = copy.deepcopy(op)      # do not disturb!
    pkeys = list(op.keys())
    pkey = pkeys[np.random.randint(len(pkeys))]



    
    if pkey == 'learnrate':
        ## value is stored directly in the dictionary
        mu = np.log(op[pkey])
        op[pkey] = np.random.lognormal(mu)

    else:
        ## Value is stored as a tuple
        val = list(op[pkey])
        if pkey == 'regularization' and np.random.randint(10) == 0:
            ## swap the regularization type
            if val[0] == 'L1':
                val[0] = 'L2'
            else:
                val[0] = 'L1'

        else:
            mu = np.log(val[1])
            val[1] = np.random.lognormal(mu)

        op[pkey] = tuple(val)

    return op

            
def genpool(parents, npool, nmutate):
    """Generate a pool of networks from a collection of parent networks.

    :param parents: List of prospective parent networks.
    :param npool: number of networks to generate for the pool
    :param nmutate: number of mutations to apply to each hybrid network

    """

    netselect = np.random.randint(len(parents), size=(npool,2))

    return [conjugate(parents[sel[0]], parents[sel[1]], nmutate) for sel in netselect]


def run_hypersearch(args):
    """Run a hyperparameter search for a specified family of networks.

    :param args: dictionary of arguments (see below)
    :return: list of triples (idx, performance, savefile)

    The arguments to be supplied in args are:
        * baseconfig  : model specification that will serve as the prototype for
                        the networks to be tested
        * inputdata   : name of the file containing the input data
        * recorddir   : name of the directory to record the results in
        * nkeep       : number of top performing models to keep
        * ngen        : number of generations to run
        * npool       : number of models to evaluate in each generation
        * nspawn      : number of models to participate in hybridization in each
                        generation
        * nmutate     : number of mutations to apply to each hybrid
        * nepoch      : number of epochs to train each model in evaluation
        * dt          : minimum time (in minutes) between index writes (default = 10)

    """

    ## TODO: other options that would be nice to be able to set:
    ##    initmutate (see below)
    ##    clobber/noclobber
    ##    minibatch size
    
    baseconfig = args['baseconfig']
    inputdata = args['inputdata']
    recorddir = args['recorddir']
    nkeep = args['nkeep']
    ngen = args['ngen']
    npool = args['npool']
    nspawn = args['nspawn']
    nmutate = args['nmutate']
    nepoch = args['nepoch']
    dt = args.get('dt', 10)

    initmutate = 2*nmutate      # number of mutations to use in the initial pool
    
    ## Create the evaluation pool by adding the base config and filling
    ## out the rest with hybrids.  
    pool = genpool([baseconfig], npool, initmutate) 

    ## List of the networks to keep (none yet, obviously)
    keepnets = []

    rr = dnnclim.RunRecorder(recorddir)
    sortkey = rr.make_sortkey()

    ## load the data and standardize it
    infile = open(inputdata, 'rb')
    climdata = pickle.load(infile)
    infile.close()
    stdfac = dnnclim.standardize(climdata)


    ## Function that does the work.  It closes over rr (the run
    ## recorder) and climdata.  If we ever want to parallelize the map
    ## call below, we will probably need to convert this into a class
    ## with a __call__ method so that we can pickle it and send it
    ## over the wire to the worker nodes.
    def run_pool_member(config, idx):
        """Run a config and return a tuple of (performance, savefile)"""
        (sf, of) = rr.filenames(idx)
        try:
            (perf, ckptfile, niter) = dnnclim.runmodel(config, climdata, stdfac=stdfac, epochs=nepoch,
                                                       savefile=sf, outfile=of, quiet=True)
        except Exception as e:
            sys.stderr.write("###Error running model: {}\n".format(e))
            sys.stderr.write("###Config: {}\n".format(config))
            perf = [9.99e99, 9.99e99]
            ckptfile = "/dev/null"
            
        return (perf, ckptfile)

    clean = True             # indicator of whether the disk copy of the index is up to date
    for gen in range(ngen):
        sys.stdout.write('Generation: {}\n'.format(gen))
        if len(pool) == 0:
            ## On the first iteration the pool will be full of models
            ## from the initialization procedure.  On iterations after
            ## the first, the eval pool will have been emptied by the
            ## previous iteration.
            pool = dnnclim.genpool(keepnets[:nspawn], npool, nmutate)

        indices = [rr.newrun(config) for config in pool]

        ## This piece is where the heavy lifting happens.  It could be
        ## parallelized over multiple nodes, but it appears that
        ## python does not have an easy equivalent of R's parmapply,
        ## so getting parallelism to work will probably take a bit
        ## more effort than we want to expend right now.
        rslts = map(run_pool_member, pool, indices)

        for (idx,rslt) in zip(indices,rslts):
            rr.record_rslts(idx, rslt[0], rslt[1])

        ## sort the pool in ascending order by performance
        ## (performance is MAE, so lower is better).  Drop any excess
        ## models from the end of the list.
        keepnets += pool
        keepnets.sort(key=sortkey)
        keepnets = keepnets[:nkeep]

        ## reset the pool for the next iteration
        pool = []

        ## Write out the run data we've collected so far
        clean = rr.writeindex(dt)

    ## Ensure that the last round of results has been written
    if not clean:
        rr.writeindex(0)
        
    ## collect and return summary of results
    indices = rr.findidx(keepnets)
    return rr.summarize(indices)
