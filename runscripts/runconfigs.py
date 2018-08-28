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

configs = [dnnclim.seedconfigs.getconfig(7)]
    

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
