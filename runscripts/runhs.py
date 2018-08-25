
import dnnclim
import pickle
import numpy as np
import sys
import yaml

## Fix the random number seed so that our results are reproducible.
np.random.seed(8675309)

## run parameters.
## TODO: read these from the command line
npool = 10                      # size of the eval pool
nkeep = 100                     # number of networks to keep
nspawn = 5                      # number of the best networks to use to generate the next eval pool
nmutate = 2                     # number of mutations to apply in each generation
recorddir = './hpsearch'
inputdata = 'testdata/dnnclim.dat'
ngen = 10
nepoch = 50

## basic configuration
baseconfig = (# config 1
    (
        ( # stage 1
            ('C', 16, (3,3)),
            ('C', 16, (3,3)),
            ('D', (2,3))
        ),
    ),
    (
        ( # stage 1
            ('U', 16, (3,3), (4,4)),
            ('C', 16, (3,3))    
        ),
        ( # stage 2
            ('U', 16, (3,3), (4,4)),
            ('C', 16, (3,3)),
        ),
        ( # stage 3
            ('U', 16, (3,3), (6,6)),
            ('C', 16, (3,3)),
        )
    ),
    (
        ( # stage 1
            ('U', 16, (3,3), (2,3)),
            ('C', 8, (3,3)),
            ('C', 2, (3,3))
        ),
    ),
    {'learnrate':0.01, 'regularization':('L1', 1.0), 'temp-loss':('normal',1.0), 'precip-loss':('qp', 1.0)}
)


## Create the evaluation pool by adding the base config and filling
## out the rest with hybrids.
pool = dnnclim.genpool([baseconfig], npool, 3) 


## List of the networks to keep, none yet obviously
keepnets = []

rr = dnnclim.RunRecorder(recorddir)
sortkey = rr.make_sortkey()

## load the data and standardize it
infile = open(inputdata, 'rb')
climdata = pickle.load(infile)
infile.close()
stdfac = dnnclim.standardize(climdata)


## Function that does the work.  It closes over rr (the run recorder) and climdata
def run_pool_member(config, idx):
    """Run a config and return a tuple of (performance, savefile)"""
    (sf, of) = rr.filenames(idx)
    (perf, ckptfile, niter) = dnnclim.runmodel(config, climdata, stdfac=stdfac, epochs=nepoch,
                                               savefile=sf, outfile=of)
    return (perf, ckptfile)

for gen in range(ngen):
    sys.stdout.write('Generation: {}\n', gen)
    if len(pool) == 0:
        ## On iterations after the first, the eval pool will have been
        ## emptied by the previous iteration
        pool = dnnclim.genpool(keepnets[:nspawn], npool, nmutate)
        
    indices = [rr.newrun(config) for config in pool]
    
    ## This piece could be parallelized
    rslts = map(run_pool_member, pool, indices)
    
    for (idx,rslt) in zip(indices,rslts):
        rr.record_rslts(idx, rslt[0], rslt[1])

    ## sort the pool in ascending order by performance (performance is
    ## MAE, so lower is better)
    keepnets += pool
    keepnets.sort(key=sortkey)
    if len(keepnets) > nkeep:
        keepnets[nkeep:] = []

    ## reset the pool for the next iteration
    pool = []

    ## Write out the run data we've collected so far
    rr.writeindex()
    

## write the best networks we found to stdout.
indices = rr.findconfig(keepnets)
sys.stdout.write('Index\tperf\tsavefile\n')
for idx in indices:
    sys.stdout.write('{}\t{}\n'.format(idx, rr.runs[idx]['lossval'], rr.runs[idx]['final-save']))

sys.stdout.write('\nFIN.\n')

        
    
