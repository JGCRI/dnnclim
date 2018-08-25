"""Tools for recording details of a set of model runs

   Typical use will be:
       * Start a new run with a specified configuration: newrun()
       * Get the names of the savefile and outfile arguments to 
         runmodel:  filenames()
       * Record the results of a run: record_rslts()
       * Write all as yet unwritten index records: writeindex()
       * Return a list of indices for a list of input configurations:
         findconfig()
"""

import yaml
import os
import numpy as np

class RunRecorder:
    """Structure for keeping and writing records of runs performed"""
    
    def __init__(self, recorddir=None, noclobber=True):
        """Initialize a run recorder

        :param recorddir: Directory to store records in.  
        :param noclobber: If true, refuese to overwrite existing output dir.
                 
        The records stored include a master index of run configurations and the
        checkpoint files generated by tensorflow. If reccorddir is omitted, the
        default is 'runrecords' in the current directory.

        """

        ## In case you're wondering, runs are stored in a dictionary
        ## rather than a list because a dictionary can be written
        ## incrementally to YAML and still produce a valid document.

        self.idx = 0            # Serial index of runs
        self.runs = {}          # dictionary of run data, indexed by serial number
        self.indices = {}       # dictionary of serial numbers, indexed by model spec
        self.unwritten = []     # Serial numbers of runs stored in self.runs but not yet written
                                #   to the index.

        if recorddir is None:
            recorddir = os.path.join(os.getcwd(), 'runrecords')
        else:
            recorddir = os.path.abspath(recorddir) 
        self.recorddir = recorddir
        self.tfsaves = os.path.join(self.recorddir, 'saves')
        self.outputs = os.path.join(self.recorddir, 'outputs')

        if not os.path.exists(self.recorddir):
            os.makedirs(self.recorddir)
            os.makedirs(self.tfsaves)
            os.makedirs(self.outputs)
        elif noclobber:
            raise IOError('Directory {} exists, and noclobber is set.'.format(self.recorddir))

        self.index = open(os.path.join(self.recorddir, 'index.yml'), 'w')
        ## TODO: write a comment with the model version at the top of the file.

    def newrun(self, modelspec):
        """Start a record for a new run

        :param modelspec: model specification for the new run
        :return: serial number for the run

        The serial number returned will be used in most of the other functions
        in this class to identify the run.

        """

        if not isinstance(modelspec, str):
            ## convert the modelspec structure to the string used to
            ## represent it.  This is the usual case, but we allow for
            ## the possibility that the user has passed in a structure
            ## that has already been converted.
            modelspec = repr(modelspec)

        ## get next available index
        idx = self.idx
        self.idx += 1

        ## TODO: There's enough going on here that the items in
        ## self.runs could probably use a class of their own.
        self.indices[modelspec] = idx
        self.runs[idx] = {}
        self.runs[idx]['modelspec'] = modelspec
        self.runs[idx]['final-save'] = None # Won't be known until we do the run
        self.runs[idx]['lossval'] = None    # Same here.


        runstr = 'save{:06d}'.format(idx) 
        outfilename = runstr + '-out.dat'
        self.runs[idx]['savebase'] = os.path.join(self.tfsaves, runstr)
        self.runs[idx]['outfile'] = os.path.join(self.outputs, outfilename)

        self.unwritten.append(idx)   # Stored but not yet written
        
        return idx

    
    def filenames(self, idxorspec):
        """Get the filename arguments for runmodel()

        :param idxorspec: Index returned by newrun(), or a modelspec
        :return: (savefilebase, outfilename)

        These filenames can be passed directly to the corresponding
        arguments for runmodel().

        """

        if isinstance(idxorspec, int):
            idx = idxorspec
        else:
            [idx] = findconfig([idxorspec])

        return (self.runs[idx]['savebase'], self.runs[idx]['outfile'])

    
    def findconfig(self, modelspecs):
        """Find the indices for a sequence of model specifications

        :param modelspecs: A sequence of model specification structures. Note
               that it has to be a sequence, even if there is only one of them.
        :return: Sequence of indices corresponding to the input model specs.

        """

        ## obviously, this won't perform gracefully if one or more of
        ## the model configurations isn't in the index.
        return [self.indices[repr(model)] for model in modelspecs]


    def record_rslts(self, idxorspec, lossval, finalsave):
        """Record the results of training a given model.

        :param idxorspec: Index returned by newrun(), or a modelspec
        :param lossval: Best value of the loss function achieved
        :param finalsave: Final save file written by tensorflow.

        """

        if isinstance(idxorspec, int):
            idx = idxorspec
        else:
            [idx] = findconfig([idxorspec])

        self.runs[idx]['lossval'] = lossval
        self.runs[idx]['final-save'] = finalsave


    def writeindex(self):
        """Write any runs recorded but not yet written into the index.

        Once you write a run into the index, there is no way to revise
        its entry, so make sure that results have been recorded for
        all outstanding runs before calling this function.

        """

        writevals = {idx:self.runs[idx] for idx in self.unwritten}

        self.index.write(yaml.dump(writevals))
        self.unwritten = []

    def make_sortkey(self):
        """Create a function that can be passed to list.sort() as a sort key.

        Handy for sorting a list of configs from best to worst.  The
        key that is used is the sum reduction of the item stored in
        the configuration's loss function.

        """

        def sortkey(config):
            [idx] = self.findconfig([config]) # TODO: really need a class for configs; too easy to forget the brackets.
            return np.sum(self.runs[idx]['lossval'])

        return sortkey
            
        
