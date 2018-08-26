#!/usr/bin/env python

import argparse
import numpy as np
import sys

#### TODO: add a way to load one or more previously trained networks
#### as a base configuration.

## Set the random number seed so that our results are reproducible.
## (Sort of.  Tensorflow has its own seed, but we can't set it until
## the tensorflow graph is set up.)
np.random.seed(8675309)


## basic configurations.  These will be selected by a command line argument.
## TODO: read these from a config file instead of defining inline.
baseconfigs = [
    (# config 1
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
    ),
]


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

argvals = parser.parse_args()

args = vars(argvals)
args['baseconfig'] = baseconfigs[argvals.configid] 

sys.stdout.write('Run options:\n')
for opt in args:
    sys.stdout.write('\t{} :  {}\n'.format(opt, args[opt]))



import dnnclim
rslts = dnnclim.run_hypersearch(args)    

sys.stdout.write('Index\tperf\tsavefile\n')
for rslt in rslts:
    sys.stdout.write('{}\t{}\t{}\n'.format(rslt[0], rslt[1], rslt[2]))

sys.stdout.write('\nFIN.\n')

        
    
