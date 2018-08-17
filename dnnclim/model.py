#!/usr/bin/env python

"""
Functions for constructing, training, end running the model

The network design is specified with a list of lists

"""

import tensorflow as tf
import numpy as np
import pickle
import os
import sys

## Number of channels in the scalar input.  Currently just Tg and t.
scalar_input_nchannel = 2

def validate_modelspec(modelspec):
    if len(modelspec) != 4:
        raise RuntimeError('validate_modelspec: expected length = 4, found length = {}'.format(len(modelspec)))

    ## Check the content of each branch
    dsbranch = modelspec[0]
    ds_stage_sizes = [(192, 288)] # size of input to each stage, also, for stage>0, size of output of previous stage.
    for stage in dsbranch:
        for layer in stage[:-1]:
            ## All layers except the last must be convolutional
            chkconv(layer, 'dsbranch')
        layer = stage[-1]
        chkmxpl(layer, 'dsbranch')

        ## Compute the output size of the stage; this will be the input size for the next stage
        insize = ds_stage_sizes[-1] 
        outsize = (int(insize[0] / layer[1][0]) , int(insize[1] / layer[1][1]))
        ds_stage_sizes.append(outsize)

    sclbranch = modelspec[1]
    scl_stage_sizes = [(1,1)]   # stage output sizes, 
    for stage in sclbranch:
        ## First layer must be an upsampling layer
        layer = stage[0]
        chkxconv(layer, 'sclbranch')
        insize = scl_stage_sizes[-1]
        outsize = (insize[0] * layer[3][0], insize[1] * layer[3][1])
        scl_stage_sizes.append(outsize)

        ## remaining layers must be convolutions
        for layer in stage[1:]:
            chkconv(layer, 'sclbranch')

    ## The output of the last stage of the scalar branch must be the
    ## same size as that of the last downsampling stage.
    if scl_stage_sizes[-1] != ds_stage_sizes[-1]:
        raise RuntimeError("Mismatch between dsbranch and sclbranch final sizes:  {} vs {}".format(ds_stage_sizes[-1], scl_stage_sizes[-1]))

    usbranch = modelspec[2]
    ds_stage_sizes.reverse()
    us_stage_sizes = ds_stage_sizes[0:1]
    for stage in usbranch:
        ## First layer must be upsampling layer
        layer = stage[0]
        chkxconv(layer, 'usbranch')
        insize = us_stage_sizes[-1]
        outsize = (insize[0] * layer[3][0], insize[1] * layer[3][1])
        us_stage_sizes.append(outsize)

        ## remaining layers must be convolutions
        for layer in stage[1:]:
            chkconv(layer, 'usbranch')

    ## The sizes of all of the stages in the upsampling branch must be
    ## the same as the corresponding stages in the downsampling
    ## branch.
    if us_stage_sizes != ds_stage_sizes:
        raise RuntimeError("Mismatch in dsbranch and usbranch stage sizes.  sizes(dsbranch):  {}  sizes(usbranch):  {}".format(ds_stage_sizes, us_stage_sizes))
        
    ## Success.  Print some summary statistics.
    sys.stdout.write('Downsampling branch:\t{} stages\tfinal size: {}\n'.format(len(dsbranch), ds_stage_sizes[0]))
    sys.stdout.write('      Scalar branch:\t{} stages\tfinal size: {}\n'.format(len(sclbranch), scl_stage_sizes[-1]))
    sys.stdout.write('  Upsampling branch:\t{} stages\tfinal size: {}\n'.format(len(usbranch), us_stage_sizes[-1]))
        
    
def build_graph(modelspec, geodata):
    """
    Given a model specification, build the tensorflow graph for that model

    :param modelspec: A model specification, as described in the documentation.
    :param geodata: Numpy array of geographical data shape = (nlat, nlon, 4).  The 4 channels are
                    lat, lon, elevation, and land fraction.
    :return: (tuple of tensors): 
             scalar input, ground truth input, model output, loss, training stepper

    """

    validate_modelspec(modelspec)

    tf.reset_default_graph()

    ## geographical data.  This is a constant.
    with tf.variable_scope('geodata'):
        geoin = tf.constant(geodata, dtype=tf.float32, shape=geodata.shape, name='geodata') 
        ## create a list to hold the inputs to each stage of the downsampling branch
        ds_stage_inputs = [tf.expand_dims(geoin, axis=0)]
    
    with tf.variable_scope('input'):
        ## scalar inputs, such as global mean temperature, time, etc.
        scalarin = tf.placeholder(dtype=tf.float32, shape=(None, scalar_input_nchannel),
                                  name='scalars')

        ## Convert scalars to a 1x1 array
        scalars = tf.expand_dims(tf.expand_dims(scalarin, axis=1), axis=2) 
        ## create a list to hold the inputs to each stage of the scalar upsampling branch
        scalar_stage_inputs = [scalars]

        ## Find out how many cases we have.  This will be needed to broadcast the geo data.
        ncase = tf.shape(scalarin)[0]

    with tf.variable_scope('groundtruth'):
        ## Real output of the ESM for evaluating loss function
        groundtruth = tf.placeholder(dtype=tf.float32, shape=(None, 192,288,2), name='groundtruth')

    ## Other arguments specification
    otherargs = modelspec[3]

    
    ## create the downsampling branch for the topo
    ## TODO: add regularization to each layer, if requested.
    dsspec = modelspec[0]
    ds_stage_outputs = []       # The last output of a stage *before* max pooling
    with tf.variable_scope('downsampling'):
        for stage in dsspec:
            layerin = ds_stage_inputs[-1]
            for layer in stage:
                if layer[0] == 'C':
                    ## convolution layer: output of the convolution is the input for the next layer.
                    layerin = mk_convlayer(layer, layerin)
                elif layer[0] == 'D':
                    ## This is the end of the stage.  Add the layer's
                    ## input to the stage output (remember, they are
                    ## recorded *before* pooling), and add the result
                    ## of the pooling to the end of the stage inputs
                    ## list.
                    ds_stage_outputs.append(layerin) # 'outputs' has the penultimate result from each stage
                    next_stage_input = mk_downsamplelayer(layer, layerin)
                    ds_stage_inputs.append(next_stage_input)
                    break
                else:
                    ## shouldn't be able to get here
                    raise RuntimeError("Invalid modelspec slipped through somehow.")

        ## The result of this branch is the last item in "stage
        ## inputs" (it's what would have been the input to the next
        ## stage, if there were one).  Broadcast it along the 0th axis
        ## to be compatible with the rest of the input data.
        dsrslt = bcast_case(ds_stage_inputs[-1], ncase)

        ## Also broadcast the tensors in the outputs arrays.
        ds_stage_outputs = [bcast_case(x, ncase) for x in ds_stage_outputs]


    ## create the scalar upsampling branch
    sclspec = modelspec[1]
    with tf.variable_scope('scalar_upsampling'):
        for stage in sclspec:
            layerin = scalar_stage_inputs[-1]
            for layer in stage:
                if layer[0] == 'U':
                    ## Upsample with transpose convolution. Output is input for next layer
                    layerin = mk_xconvlayer(layer, layerin)
                elif layer[0] == 'C':
                    layerin = mk_convlayer(layer, layerin)
                else:
                    ## should't be able to get here
                    raise RuntimeError("Invalid modelspec slipped through somehow.")

            ## output from last layer of the stage becomes input for next stage
            scalar_stage_inputs.append(layerin)


    ## concatenate the result of the downsampling to the result of the
    ## scalar upsampling to form the input for the upsampling branch.
    ## The final result of each branch is the last element in its list
    ## of stage inputs.
    with tf.variable_scope('join_ds2scl'):
        us_stage_inputs = [tf.concat([dsrslt, scalar_stage_inputs[-1]],
                                     axis = 3,
                                     name='join_ds_to_scalar')]

    ## create the main upsampling branch
    usspec = modelspec[2]
    ds_stage_idx = 0
    ds_stage_outputs.reverse()

    with tf.variable_scope('upsampling'):
        for stage in usspec:
            layerin = us_stage_inputs[-1]

            for layer in stage:
                if layer[0] == 'U':
                    ## Upsample.  Then bind the last tensor from the
                    ## corresponding stage of the downsampling branch.
                    upsamp = mk_xconvlayer(layer, layerin) 
                    bindlayer = ds_stage_outputs[ds_stage_idx]
                    ds_stage_idx += 1

                    tf.assert_equal(tf.shape(bindlayer)[0:3],
                                    tf.shape(upsamp)[0:3],
                                    summarize=3,
                                    message='Incompatible shapes joining U branches: ')
                    layerin = tf.concat([bindlayer, upsamp], axis=3, name='join_U_branch')
                    # layerin = upsamp
                elif layer[0] == 'C':
                    layerin = mk_convlayer(layer, layerin)
                else:
                    ## should be able to get here
                    raise RuntimeError("Invalid modelspec slipped through somehow")
        output = layerin        # i.e., the result of the last convolutional layer in the last stage
        
    ## That's all of the branches, now extract the results.
    
    ## TODO: Do we need some additional processing of the last layer
    ## in the upsampling branch?  For example, we might want to
    ## convert temperature to temperature anomaly or apply an
    ## exponential transform to precip to ensure it is >= 0.
    
    ## Calculate a loss function.  For now we will just use RMS error,
    ## though that may not be the right loss to use with
    ## precipitation.  Consider using a generalized Poisson loss given
    ## by lgamma(x, xtrue) - lgamma(x,0) (where lgamma is the log of
    ## the incomplete gamma funciton.  It looks like you could you
    ## could compute this as -tf.log(tf.igammac(x, xtrue)).

    with tf.variable_scope('loss_calculation'):
        tf.assert_equal(tf.shape(groundtruth),
                        tf.shape(output))
        loss = tf.reduce_sum(tf.squared_difference(output, groundtruth))

    train_step = tf.train.AdamOptimizer(otherargs['learnrate']).minimize(loss, name='train_step')

    return(scalarin, groundtruth, output, loss, train_step)

def bcast_case(tensorin, ncase):
    """Broadcast case-independent tensor to be compatible with case-dependent tensors."""
    newshape = tf.concat([(ncase,), tf.shape(tensorin)[1:]], axis=0)
    return tensorin + tf.zeros(newshape)


def runmodel(modelspec, climdata, savefile=None):
    """Train and evaluate a model.

    :param modelspec: A model specification structure
    :param climdata: Structure containing training and dev data sets
    :param savefile: Base filename for checkpoint outputs
    :return: (perf, chkfile) Dev set performance and full name of the checkpoint file 
             corresponding to the best performance (i.e., including the iteration number)

    """

    (scalarin, groundtruth, output, loss, train_step) = build_graph(modelspec, climdata['geo'])

    with tf.Session() as sess:
        if savefile is not None:
            saver = tf.train.Saver()
        summarywriter = tf.summary.FileWriter('logs', sess.graph)

        init = tf.global_variables_initializer()
        sess.run(fetches=[init])

        fd={scalarin:climdata['train']['gmean'],
            groundtruth:climdata['train']['fld']}
        (lossval,) = sess.run(fetches=[loss], feed_dict=fd)



        return (lossval, None)


#### Helper functions

def mk_convlayer(spec, layerin):
    """Make a convolutional layer from its specification"""
    
    nfilt = spec[1]
    dimfilt = spec[2]
    return tf.layers.conv2d(layerin, nfilt, dimfilt,
                            padding='SAME', activation=tf.nn.relu)


def mk_downsamplelayer(spec, layerin):
    """Make a downsampling (max pooling) layer from its specification"""
    
    dimpool = spec[1]
    return tf.layers.max_pooling2d(layerin, pool_size=dimpool, strides=dimpool,
                                   padding='SAME')


def mk_xconvlayer(spec, layerin):
    nfilt = spec[1]
    dimfilt = spec[2]
    stride = spec[3]
    return tf.layers.conv2d_transpose(layerin, nfilt, dimfilt, stride,
                                      padding='SAME', activation=tf.nn.relu)
    

def chkconv(layer, brname):
    """Check that the layer specification for a convolutional layer is valid.

    Valid format:  ('C', nfilt, (filt_xsize, filt_ysize))

    """

    if layer[0] != 'C' or len(layer) != 3 or not isinstance(layer[1], int)  or not chkintseq(layer[2], 2):
        raise RuntimeError('validate_modelspec: illegal layer in {}. Found spec: "{}"'.format(brname, layer))

def chkmxpl(layer, brname):
    """Check that the layer specification for a max pooling layer is valid.

    Valid format: ('D', (poox_xdim, pool_ydim))

    """

    if layer[0] != 'D' or len(layer) != 2 or not chkintseq(layer[1], 2):
        raise RuntimeError('validate_modelspec: illegal layer in {}. Found spec: "{}"'.format(brname, layer))

def chkxconv(layer, brname):
    """Check that the layer specification for a transpose convolution layer is valid.

    Valid format: ('U', nfilt, (filt_xdim, filt_ydim), (stride_x, stride_y))

    """

    if (layer[0] != 'U' or not isinstance(layer[1], int) or
        not chkintseq(layer[2],2) or not chkintseq(layer[3],2)):
        raise RuntimeError('chkxconv: illegal layer in {}. Found spec: {}'.format(brname, layer))


def chkintseq(seq, exlen):
    """Check that an object is a sequence of integers with specified length"""

    if len(seq) == exlen and all([isinstance(x, int) for x in seq]):
        return True
    else: 
        return False



#### TODO:
####   Calculate number of parameters
####   Add geographic coords as inputs.
