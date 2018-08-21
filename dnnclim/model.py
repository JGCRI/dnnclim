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
geo_input_nchannel = 4
geo_input_imgsize = (192,288)

## We use a quasipoisson likelihood function for precipitation.  In
## order for this to be remotely valid, precip must be scaled so that
## its variance is equal (more or less) to its mean.  This factor
## scales the precip field so that the likely model-specified scale
## factor should be close to 1.  It is calculated as the median over
## all grid cells in the training set of the ratio of the per-cell
## variance to the per-cell mean.
precip_intrinsic_scale = 4e-7

def validate_modelspec(modelspec):
    """Check to see that a model specification meets all of the, and count number of parameters.

    :param modelspec: A model specification structure
    :return: number of parameters in the model

    """

    if len(modelspec) != 4:
        raise RuntimeError('validate_modelspec: expected length = 4, found length = {}'.format(len(modelspec)))

    chkotherargs(modelspec[3]) 
    
    pcount = 0                  # running count of free parameters
    
    ## Check the content of each branch
    dsbranch = modelspec[0]
    ds_stage_sizes = [(192, 288, geo_input_nchannel)] # size of input to each stage, also, for stage>0, size of output of previous stage.
    ds_copy_sizes = []                                 # size of the images that will be copied to the upsampling branch
    dsnchannel = geo_input_nchannel # number of channels coming into each layer
    for stage in dsbranch:
        for layer in stage[:-1]:
            ## All layers except the last must be convolutional
            chkconv(layer, 'dsbranch')
            pcount += convnparam(layer, dsnchannel)
            print('***layer param: {}'.format(convnparam(layer, dsnchannel)))
            print('\tnchannel in= {}  nchannel out= {}'.format(dsnchannel, layer[1]))
            dsnchannel = layer[1]
        layer = stage[-1]
        chkmxpl(layer, 'dsbranch') # no parameters for max pooling layer.

        ## Compute the output size of the stage; this will be the input size for the next stage
        insize = ds_stage_sizes[-1]
        ds_copy_sizes.append((insize[0], insize[1], dsnchannel))
        outsize = (int(insize[0] / layer[1][0]) , int(insize[1] / layer[1][1]), dsnchannel)
        ds_stage_sizes.append(outsize)

    sclbranch = modelspec[1]
    scl_stage_sizes = [(1,1,scalar_input_nchannel)]   # stage output sizes,
    sclnchannel = scalar_input_nchannel
    for stage in sclbranch:
        ## First layer must be an upsampling layer
        layer = stage[0]
        chkxconv(layer, 'sclbranch') 
        pcount += convnparam(layer, sclnchannel) # convnparam also calculates nparam for xposeconv
        print('***layer param: {}'.format(convnparam(layer, sclnchannel)))
        sclnchannel = layer[1]

        insize = scl_stage_sizes[-1]
        outsize = (insize[0] * layer[3][0], insize[1] * layer[3][1], sclnchannel)
        scl_stage_sizes.append(outsize)

        ## remaining layers must be convolutions
        for layer in stage[1:]:
            chkconv(layer, 'sclbranch')
            pcount += convnparam(layer, sclnchannel)
            print('***layer param: {}'.format(convnparam(layer, sclnchannel)))
            sclnchannel = layer[1]

    ## The output of the last stage of the scalar branch must be the
    ## same size as that of the last downsampling stage.  However, the
    ## number of channels need not match.
    if scl_stage_sizes[-1][0:2] != ds_stage_sizes[-1][0:2]:
        raise RuntimeError("Mismatch between dsbranch and sclbranch final sizes:  {} vs {}".format(ds_stage_sizes[-1], scl_stage_sizes[-1]))

    usbranch = modelspec[2]
    ds_stage_sizes.reverse()
    ds_copy_sizes.reverse()
    us_stage_sizes = ds_stage_sizes[0:1]
    usnchannel = sclnchannel + dsnchannel
    dsidx = 0
    print('ds_stage_sizes: {}'.format(ds_stage_sizes))
    for stage in usbranch:
        ## First layer must be upsampling layer
        layer = stage[0]
        chkxconv(layer, 'usbranch') 
        pcount += convnparam(layer, usnchannel) # conv and xconv calculate nparam the same way
        print('***layer param: {}'.format(convnparam(layer, usnchannel)))
        usnchannel = layer[1]
        
        insize = us_stage_sizes[-1]
        outsize = (insize[0] * layer[3][0], insize[1] * layer[3][1], usnchannel)
        us_stage_sizes.append(outsize)

        ## account for the additional channels added from the ds branch
        print('\tU-bind accounting:  usnchannel= {}  dsnchannel= {}'.format(usnchannel, ds_copy_sizes[dsidx][2]))
        usnchannel += ds_copy_sizes[dsidx][2]
        dsidx += 1
        
        ## remaining layers must be convolutions
        for layer in stage[1:]:
            chkconv(layer, 'usbranch')
            pcount += convnparam(layer, usnchannel)
            print('***layer param: {}'.format(convnparam(layer, usnchannel)))
            print('\tusnchannel = {}'.format(usnchannel))
            usnchannel = layer[1]

    ## The image sizes of all of the stages in the upsampling branch
    ## must be the same as the corresponding stages in the
    ## downsampling branch. The numbers of channels don't have to
    ## match.
    for (usstage, dsstage) in zip(us_stage_sizes, ds_stage_sizes):
        if usstage[0:2] != dsstage[0:2]:
            raise RuntimeError("Mismatch in dsbranch and usbranch stage sizes.  sizes(dsbranch):  {}  sizes(usbranch):  {}".format(ds_stage_sizes, us_stage_sizes))
        
    ## Success.  Print some summary statistics.
    sys.stdout.write('Downsampling branch:\t{} stages\tfinal size: {}\n'.format(len(dsbranch), ds_stage_sizes[0]))
    sys.stdout.write('      Scalar branch:\t{} stages\tfinal size: {}\n'.format(len(sclbranch), scl_stage_sizes[-1]))
    sys.stdout.write('  Upsampling branch:\t{} stages\tfinal size: {}\n'.format(len(usbranch), us_stage_sizes[-1]))
    sys.stdout.write('\nTotal free parameters:\t{}\n'.format(pcount))

    return pcount

    
def build_graph(modelspec, geodata):
    """
    Given a model specification, build the tensorflow graph for that model

    :param modelspec: A model specification, as described in the documentation.
    :param geodata: Numpy array of geographical data shape = (nlat, nlon, 4).  The 4 channels are
                    lat, lon, elevation, and land fraction.
    :return: (tuple of tensors): 
             scalar input, ground truth input, model output, temperature loss, precip loss,
             regularization penalty, total_loss, training stepper

    The temperature and precipitation losses are the measures of discrepancy for the 
    temperature and precipitation variables.  The quantity being optimized is the 
    "total loss", which is the sum of the two, plus the regularization penalty.

    """

    nparam = validate_modelspec(modelspec)

    tf.reset_default_graph()

    ## select regularization to add to the layers.
    regspec = modelspec[3]['regularization']
    ## Rescale the regularization scale by the ratio of image size to
    ## number of parameters.  This helps ensure that the amount of
    ## regularization required to make a meaningful contribution to
    ## the total loss is not so sensitive to the number of parameters.
    ## Our calculation of the number of parameters isn't perfect for
    ## this purpose, since it includes the bias values, which aren't
    ## being regularized, but it should be close enough.
    regscl = regspec[1] * np.prod(geodata.shape) / nparam
    print('regscl = {}'.format(regscl))
    if regspec[0] == 'L1':
        reg = lambda: tf.contrib.layers.l1_regularizer(regscl)
    elif regspec[0] == 'L2':
        reg = lambda: tf.contrib.layers.l2_regularizer(regscl)
    else:
        reg = lambda: None
        
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
                    layerin = mk_convlayer(layer, layerin, reg)
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
                    layerin = mk_xconvlayer(layer, layerin, reg)
                elif layer[0] == 'C':
                    layerin = mk_convlayer(layer, layerin, reg)
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
                    upsamp = mk_xconvlayer(layer, layerin, reg) 
                    bindlayer = ds_stage_outputs[ds_stage_idx]
                    ds_stage_idx += 1

                    tf.assert_equal(tf.shape(bindlayer)[0:3],
                                    tf.shape(upsamp)[0:3],
                                    summarize=3,
                                    message='Incompatible shapes joining U branches: ')
                    layerin = tf.concat([bindlayer, upsamp], axis=3, name='join_U_branch')
                    # layerin = upsamp
                elif layer[0] == 'C':
                    layerin = mk_convlayer(layer, layerin, reg)
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
        (temploss, preciploss) = mk_losscalc(groundtruth, output, otherargs)
        loss = temploss + preciploss
        reg_pen = tf.losses.get_regularization_loss()
        total_loss = loss + reg_pen

    train_step = tf.train.AdamOptimizer(otherargs['learnrate']).minimize(total_loss, name='train_step')

    return(scalarin, groundtruth, output, temploss, preciploss, reg_pen, total_loss, train_step)

def bcast_case(tensorin, ncase):
    """Broadcast case-independent tensor to be compatible with case-dependent tensors."""
    newshape = tf.concat([(ncase,), tf.shape(tensorin)[1:]], axis=0)
    return tensorin + tf.zeros(newshape)


def runmodel(modelspec, climdata, epochs=100, batchsize=15, savefile=None):
    """Train and evaluate a model.

    :param modelspec: A model specification structure
    :param climdata: Structure containing training and dev data sets
    :param savefile: Base filename for checkpoint outputs
    :return: (perf, chkfile, count) Dev set performance, full name of the checkpoint file 
             corresponding to the best performance (i.e., including the iteration number), 
             and total number of epochs run.

    """

    (scalarin, groundtruth, output, temploss, preciploss,
     reg_pen, total_loss, train_step) = build_graph(modelspec, climdata['geo'])

    if savefile is not None:
        ckptr = tf.train.Saver()
        epochckpt = -1          # epoch when the model was last checkpointed
        ## factor to convert total error to mean error per grid cell.
        fldsize = climdata['dev']['fld'].shape
        normfac = 2.0/np.prod(fldsize) # factor of 2 because we want
                                       # to normalize each of the 2
                                       # fields separately.
    
    with tf.Session() as sess:
        summarywriter = tf.summary.FileWriter('logs', sess.graph)

        init = tf.global_variables_initializer()
        sess.run(fetches=[init])

        ## pull the training set.  'x' is the input (global means,
        ## time, etc), and 'y' is the output (the fields produced by
        ## the ESMs).  The geo data was set up as constants when we
        ## built the graph.
        train_x = climdata['train']['gmean']
        train_y = climdata['train']['fld']
        ntrain = train_x.shape[0]

        ## pull the dev set.  We'll use that to monitor our progress.
        dev_x = climdata['dev']['gmean']
        dev_y = climdata['dev']['fld']
        min_loss = np.inf

        patience = 10
        patcount = 0
        
        for epoch in range(epochs):
            ## shuffle the data once per epoch
            idx = np.random.permutation(ntrain)
            train_x = train_x[idx,:]
            train_y = train_y[idx,:]

            ## run each minibatch
            for update in range(int(np.floor(ntrain/batchsize))):
                mbidx = np.arange(update * batchsize, (update+1) * batchsize)
                mb_x = train_x[mbidx,:]
                mb_y = train_y[mbidx,:]

                fd={scalarin:mb_x, groundtruth:mb_y}
                (_,) = sess.run(fetches=[train_step], feed_dict=fd)

            ## at the end of each epoch, evaluate on the dev set.  If
            ## we've improved on the previous best value, record the
            ## model in a checkpoint.
            fd={scalarin:dev_x, groundtruth:dev_y}
            fetch = [temploss, preciploss, total_loss, reg_pen]
            (ltemp, lprecip, ltot, regval) = sess.run(fetches=fetch, feed_dict=fd)
            if ltot < min_loss:
                min_loss = ltot
                patcount = 0
                if savefile is not None:
                    ckptr.save(sess, savefile, global_step=epoch)
                    epochckpt = epoch
                    tempnorm = ltemp * normfac
                    precipnorm = lprecip * normfac
                    regnorm = regval * normfac
                    totalnorm = ltot * normfac
                    outstr = 'Model checkpoint at epoch= {}, \n\ttemploss per grid cell= {},  preciploss per grid cell= {}  \n\tregval= {}  totalloss= {}\n'
                    sys.stdout.write(outstr.format(epoch,tempnorm, precipnorm, regnorm, totalnorm))
            else:
                patcount += 1

            if patcount >= patience:
                break


    if savefile is not None:
        saveout = '{}-{}'.format(savefile, epochckpt)
    else:
        saveout = None
    return (ltot, saveout, epoch)


#### Helper functions

def mk_losscalc(obs, model, oparam):
    """Add nodes to the graph to calculate the loss function.
    :param obs: tensor(N,nlat,nlon,nvar) of the observed data (i.e., the data we are 
                trying to model.)
    :param model: tensor(N, nlat, nlon, nvar) of the model output.
    :param oparam: The other-params structure from the model specification.  This will
                   contain the likelihood function definitions for each of the variables. 
    :return: tuple of loss function values, calculated as the negative
             of the total log-likelihood for the data for each
             variable.

    For the time being, nvar is 2, temperature and precipitation.
    These two variables behave very differently, and so it's important
    to use a likelihood function that reflects the expected
    distribution of each.  Furthermore, each likelihood function will
    have one or more parameters, which are specified in the oparam
    argument.

    """

    ## split data into temperature and precip
    tempobs = obs[:,:,:,0]
    tempmod = model[:,:,:,0]

    precipobs = obs[:,:,:,1]
    precipmod = model[:,:,:,1]

    temploss = mk_normal_loss(tempobs, tempmod, oparam['temp-loss'])
    preciploss = mk_qp_loss(precipobs, precipmod, oparam['precip-loss'])

    return (temploss, preciploss)

def mk_normal_loss(obs, model, param):
    """Calculate a loss function for a normal likelihood
    :param obs: (tensor) observed data
    :param model: (tensor) model output
    :param param: (tuple) likelihood parameters.  For this likelihood function, the only
                  parameter is the scale parameter.

    The log-likelihood for a normal likelihood function is just sum(
    -(xobs-xmod)^2/(2*sig^2) ).  

    Yes, I am aware that the docstring for this function is
    considerably longer than the function it documents.

    """

    sigval = param[1]
    
    ## We have to include the sigma contribution to the normalization
    ## factor, since that's what prevents us from "improving" the fit
    ## by simply making sigma very large.
    with tf.variable_scope('normal_likelihood'):
        sig = tf.constant(2.0*sigval, name='sig', dtype=tf.float32)
        normfac = tf.constant(np.log(sigval), name='normfac', dtype=tf.float32)
        obsscl = tf.divide(obs, sig, name='norm_obsscl')
        modscl = tf.divide(model, sig, name='norm_modscl')
        return tf.reduce_sum(tf.square(obsscl-modscl) + normfac, name='normal_loss')
        #return tf.reduce_sum(tf.squared_difference(obsscl, modscl), name='normal_loss')
    


def mk_qp_loss(obs, model, param):
    """Calculate a quasi-poisson loss
    :param: obs: (tensor) observed data
    :param model: (tensor) model output
    :param param: (tuple) likelihod parameters.  The only parameter for this likelihood function
                  is a scale factor.

    The log-likelihood function for the quasi-poisson likelihood function is
    -mod/sig + obs/sig * log(mod/sig) - lgamma(1+obs/sig).  From this you must subtract
    the likelihood of the "saturated model", which is a hypothetical model that perfectly
    captures the data:  -obs/sig * (1 - log(obs/sig))

    What?  You're worried about that log because obs could be zero?  Don't be.  The limit as
    obs approaches zero is zero.  In practice, we cut obs off at some small value where the 
    result is close enough to zero for our purposes.
    """

    ## protect against undefined values.  The limits are all well
    ## defined at zero, so it's fine to just cut off the inputs at
    ## some sufficiently small fixed value.
    with tf.variable_scope('qp_likelihood'):
        sig = param[1]*precip_intrinsic_scale
        print('qp sig= {}', sig)
        sigfac = tf.constant(1.0/sig, name='sclfac', dtype=tf.float32)
        minval = 1.0e-8
        obsscl = tf.maximum(minval, obs*sigfac, name='obsscl')
        modscl = tf.maximum(minval, model*sigfac, name='modscl')

        ## recall loss = -loglik
        logdenom = tf.lgamma(1.0+obsscl, name='logdenom')
        with tf.variable_scope('model_term'):
            modloss = modscl - obsscl*tf.log(modscl) + logdenom
        with tf.variable_scope('saturated_term'):
            satloss = obsscl - obsscl*tf.log(obsscl) + logdenom

        return tf.reduce_sum(modloss - satloss, name='qp_loss')
    

def mk_convlayer(spec, layerin, mkreg):

    """Make a convolutional layer from its specification"""
    
    nfilt = spec[1]
    dimfilt = spec[2]
    return tf.layers.conv2d(layerin, nfilt, dimfilt,
                            padding='SAME', activation=tf.nn.relu,
                            kernel_regularizer=mkreg())


def mk_downsamplelayer(spec, layerin):
    """Make a downsampling (max pooling) layer from its specification"""
    
    dimpool = spec[1]
    return tf.layers.max_pooling2d(layerin, pool_size=dimpool, strides=dimpool,
                                   padding='SAME')


def mk_xconvlayer(spec, layerin, mkreg):
    nfilt = spec[1]
    dimfilt = spec[2]
    stride = spec[3]
    return tf.layers.conv2d_transpose(layerin, nfilt, dimfilt, stride,
                                      padding='SAME', activation=tf.nn.relu,
                                      kernel_regularizer=mkreg())
    

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


def chkotherargs(otherargs):
    """Check that the otherargs section of the model spec is valid

    This section must be a dictionary containing the following entries:
        * learnrate: float > 0
        * regularization: tuple (type, scale).  Type is 'L1', 'L2', or 'N' (for none).
                          Scale is float > 0.
    """

    if 'learnrate' not in otherargs:
        raise RuntimeError('Config must supply learning rate.')
    elif otherargs['learnrate'] <= 0:
        raise ValueError('learnrate must be > 0.')

    if 'regularization' not in otherargs:
        raise RuntimeError('Config must specify regularization type and scale.')

    reg = otherargs['regularization']
    if reg[0] not in ('L1', 'L2', 'N'):
        raise ValueError('Invalid regularization type. Valid types are L1, L2, N. Found {}'.format(reg[0]))
    elif reg[1] < 0:
        raise ValueError('Regularization scale must be >= 0.  Found {}'.format(reg[1]))

    
def convnparam(layer, nchannel, bias=True):
    """Calculate number of parameters in a convolutional or transpose convolutional layer

    :param layer: (tuple) layer specification
    :param nchannel: (int) Number of channels input to the layer
    :param bias: (bool) Whether or not the layer includes a bias before the activation
    :return: (int) number of parameters contributed by the layer

    Convolutions and transpose convolutions calculate their number of
    parameters the same way, so this function serves for both.

    """

    nfilt = layer[1]
    (nx, ny) = layer[2]
    np = nfilt*nx*ny*nchannel

    if bias:
        np += nfilt

    return np


#### TODO:
####   Calculate number of parameters
####   Add geographic coords as inputs.
