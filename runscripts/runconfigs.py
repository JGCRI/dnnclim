#!/usr/bin/env python

import dnnclim
import pickle
import sys
import time

if len(sys.argv) > 1:
    rundir = sys.argv[1]
else:
    rundir = 'test'

if len(sys.argv) > 2:
    mbsize = int(sys.argv[2])
else:
    mbsize = 15

configs = [
    (# config 0
        (
            ( # stage 1
                ('C', 10, (3,3)),
                ('C', 10, (3,3)),
                ('D', (2,3))
            ),
        ),
        (
            ( # stage 1
                ('U', 10, (3,3), (4,4)),
                ('C', 10, (3,3))    
            ),
            ( # stage 2
                ('U', 10, (3,3), (4,4)),
            ),
            ( # stage 3
                ('U', 10, (3,3), (6,6)),
            )
        ),
        (
            ( # stage 1
                ('U', 10, (3,3), (2,3)),
                ('C', 2, (3,3))
            ),
        ),
        {'learnrate':0.01, 'regularization':('L1', 1.0), 'temp-loss':('normal',1.0), 'precip-loss':('qp', 1.0)}
    ),

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
    

infile = open('testdata/dnnclim.dat','rb')
climdata = pickle.load(infile)
infile.close()

rr = dnnclim.RunRecorder(rundir, noclobber=False)

for config in configs:
    idx = rr.newrun(config)
    (sf, of) = rr.filenames(idx)
    t1 = time.time()
    (loss, ckptfile, niter) = dnnclim.runmodel(config, climdata, epochs=25, savefile=sf, outfile=of,
                                               batchsize=mbsize)
    t2 = time.time()
    eff = (t2-t1) / niter
    print('Finished run {} in {} iterations. outfile= {}, loss= {}\t{} seconds/epoch'.format(idx, niter, of, loss, eff))
    rr.record_rslts(idx, loss, ckptfile)

rr.writeindex()

print('FIN.')
