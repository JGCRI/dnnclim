#!/usr/bin/env python

import argparse
import numpy as np
import sys
import dnnclim.seedconfigs

#### TODO: add a way to load one or more previously trained networks
#### as a base configuration.

## Set the random number seed so that our results are reproducible.
## (Sort of.  Tensorflow has its own seed, but we can't set it until
## the tensorflow graph is set up.)
np.random.seed(8675309)


parser = argparse.ArgumentParser()

## TODO: add a clobber argument
parser.add_argument('inputdata', help='Filename for the input data')
parser.add_argument('recorddir', help='Directory to write results into. Must *not* already exist')
parser.add_argument('configid', type=int, help='Index in the list of base configs')
parser.add_argument('-k', dest='nkeep', type=int, help='Number of models to keep', default=100)
parser.add_argument('-g', dest='ngen', type=int, help='Number of generations to run', default=10)
parser.add_argument('-p', dest='npool', type=int, help='Number of models in the eval pool', default=10)
parser.add_argument('-s', dest='nspawn', type=int, help='Number of models to participate in hybridization',
                    default=10)
parser.add_argument('-m', dest='nmutate', type=int, help='Number of mutations to apply to each hybrid model',
                    default=1)
parser.add_argument('-e', dest='nepoch', type=int, help='Number of epochs to train each model when evaluating',
                    default=1000)
parser.add_argument('-dt', type=int, help='Minimum time between data dumps to disk (in minutes)', default=10)

argvals = parser.parse_args()

args = vars(argvals)
args['baseconfig'] = dnnclim.seedconfigs.getconfig(argvals.configid)

sys.stdout.write('Run options:\n')
for opt in args:
    sys.stdout.write('\t{} :  {}\n'.format(opt, args[opt]))

import dnnclim
dnnclim.model.write_summaries = False
sys.stdout.write('\nSeed model summary:\n')
dnnclim.model.config_report(args['baseconfig'], sys.stdout)

rslts = dnnclim.run_hypersearch(args)    

sys.stdout.write('Index\tperf\tsavefile\n')
for rslt in rslts:
    sys.stdout.write('{}\t{}\t{}\n'.format(rslt[0], rslt[1], rslt[2]))

sys.stdout.write('\nFIN.\n')

        
    
