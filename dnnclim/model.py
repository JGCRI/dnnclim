#!/usr/bin/env python

"""
Functions for constructing, training, end running the model

The network design is specified with a list of lists

"""

import tensorflow as tf
import numpy as np
import pickle
import os

## Number of channels in the scalar input.  Currently just Tg and t.
scalar_input_nchannel = 2

def validate_modelspec(modelspec):
    ## TODO:  implement model checks.
    return True

def build_graph(modelspec):
    """
    Given a model specification, build the tensorflow graph for that model

    :param modelspec: A model specification, as described in the documentation.
    :return: (tuple of tensors): 
             topo input, scalar input, ground truth input, model output, loss, training stepper

    """

    validate_modelspec(modelspec)

    tf.reset_default_graph()

    with tf.variable_scope('input'):
        ## topo data.  This is the same for all cases.
        topoin = tf.placeholder(dtype=tf.float32, shape=(192,288,2), name='topo')
        ## scalar inputs, such as global mean temperature, time, etc.
        scalarin = tf.placeholder(dtype=tf.float32, shape=(None, scalar_input_nchannel),
                                  name='scalars')
        ## Real output of the ESM for evaluating loss function
        groundtruth = tf.placeholder(dtype=tf.float32, shape=(None, 192,288,2), name='groundtruth')

        ## Convert scalars to a 1x1 array
        scalars = tf.expand_dims(tf.expand_dims(scalarin, axis=1), axis=2)

        ## create a list to hold the inputs to each stage of each branch.
        ds_stage_inputs = [topoin]
        scalar_stage_inputs = [scalars]

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
                    dimpool = layer[1]
                    ds_stage_outputs.append(layerin)
                    ds_stage_inputs.append(mk_downsamplelayer(layer, layerin))
                    break
                else:
                    ## shouldn't be able to get here
                    raise "Invalid modelspec slipped through somehow."

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
                    raise "Invalid modelspec slipped through somehow."

            ## output from last layer of the stage becomes input for next stage
            scalar_stage_inputs.append(layerin)

    ## concatenate the result of the downsampling to the result of the
    ## scalar upsampling to form the input for the upsampling branch.
    ## The final result of each branch is the last element in its list
    ## of stage inputs.
    tf.assert_equal(tf.shape(ds_stage_inputs[-1])[0:3],
                    tf.shape(scalar_stage_inputs[-1])[0:3],
                    summarize=3,
                    message='Incompatible shapes joining scalars to downsampling')
    us_stage_inputs = [tf.concat([ds_stage_inputs[-1], scalar_stage_inputs[-1]],
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
                    layerin = tf.concat([bindlayer, layerin], axis=3, name='join_U_branch')
                elif layer[0] == 'C':
                    layerin = mk_convlayer(layer, layerin)
                else:
                    ## should be able to get here
                    raise "Invalid modelspec slipped through somehow"
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

    return(topoin, scalarin, groundtruth, output, loss, train_step)
    



def mk_convlayer(spec, layerin):
    """Make a convolutional layer from its specification"""
    
    nfilt = spec[1]
    dimfilt = spec[2]
    return tf.layers.conv2d(layerin, nfilt, dimfilt, padding='SAME')


def mk_downsamplelayer(spec, layerin):
    """Make a downsampling (max pooling) layer from its specification"""
    
    dimpool = layer[1] 
    return tf.layers.max_pooling2d(layerin, dimpool, strides=(1,1), padding='SAME')


def mk_xconvlayer(spec, layerin):
    nfilt = layer[1]
    dimfilt = layer[2]
    stride = layer[3]
    return tf.layers.conv2d_transpose(layerin, nfilt, dimfilt, stride, padding='SAME')
    
