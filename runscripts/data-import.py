#!/usr/bin/env python 


# ## Import data from the current directory and write to default dump file

# In[1]:

from dnnclim import data
import sys

if len(sys.argv) < 2:
    print('Usage: {} <outfilename>'.format(argv[0]))
    sys.exit(0)

outfilename = sys.argv[1]
    

climdata = data.process_dir('.', outfile=outfilename)


# ### Print field sizes

for dset in climdata:
    print('dataset: {}'.format(dset))
    if dset == 'geo':
        print('\tgeo: ', climdata[dset].shape)
    else:
        print('\tfld: ', climdata[dset]['fld'].shape)
        print('\tgmean: ', climdata[dset]['gmean'].shape)

