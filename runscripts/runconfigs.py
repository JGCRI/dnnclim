#!/usr/bin/env python

import dnnclim
import copy
import pickle

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

rr = dnnclim.RunRecorder('./test01')

for config in configs:
    idx = rr.newrun(config)
    (sf, of) = rr.filenames(idx)
    (loss, ckptfile, niter) = dnnclim.runmodel(config, climdata, epochs=1000, savefile=sf, outfile=of)
    print('Finished run {} in {} iterations. outfile= {}, loss= {}'.format(idx, niter, of, loss))
    rr.record_rslts(idx, loss, ckptfile)

rr.writeindex()

print('FIN.')
