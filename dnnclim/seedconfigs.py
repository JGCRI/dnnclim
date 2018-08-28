#!/usr/bin/env python

"""Prototype configurations for models with various numbers of stages.

    These configurations are intended to be used with the functions in
    dnnclim.hypersearch to generate candidate models to test.  The
    seed configurations aren't compatible with each other, so you
    can't hybridize them directly.

    Use dnnclim.seedconfigs.getconfig(n) to retrieve the nth
    seed configuration.

"""

_configs = [
    (# Single downsampling stage, 3 scalar upsamplings
        (
            ( # stage 1
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
            ),
        ),
        {'learnrate':0.01, 'regularization':('L1', 1.0), 'temp-loss':('normal',5.0), 'precip-loss':('qp', 1.0)}
    ),

    (# Single downsampling stage,  scalar upsamplings by factors of 2 or 3 only
        (
            ( # stage 1
                ('C', 16, (3,3)),
                ('D', (2,3))
            ),
        ),
        (
            ( # stage 1a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 1b
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 2a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3)),
            ),
            ( # stage 2b
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3)),
            ),
            ( # stage 3a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3)),
            ),
            ( # stage 3b
                ('U', 16, (3,3), (3,3)),
                ('C', 16, (3,3)),
            )            
        ),
        (
            ( # stage 1
                ('U', 16, (3,3), (2,3)),
                ('C', 8, (3,3)),
            ),
        ),
        {'learnrate':0.01, 'regularization':('L1', 1.0), 'temp-loss':('normal',5.0), 'precip-loss':('qp', 1.0)}
    ),

    (# Downsample to 48x48, upsample to match
        (
            ( # stage 1
                ('C', 16, (3,3)),
                ('D', (2,3))
            ),
            ( # stage 2
                ('C', 16, (3,3)),
                ('D', (2,2))
            ),            
        ),
        (
            ( # stage 1a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 1b
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 2a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 2b
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ), 
            ( # stage 3
                ('U', 16, (3,3), (3,3)),
                ('C', 16, (3,3)),
            )
        ),
        (
            ( # stage 1
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),
            ( # stage 2
                ('U', 16, (3,3), (2,3)),
                ('C', 8, (3,3)),
            ),            
        ),
        {'learnrate':0.01, 'regularization':('L1', 1.0), 'temp-loss':('normal',5.0), 'precip-loss':('qp', 1.0)}
    ),

    (# Downsample to 32x32, upsample to match
        (
            ( # stage 1
                ('C', 16, (3,3)),
                ('D', (2,3))
            ),
            ( # stage 2
                ('C', 16, (3,3)),
                ('D', (3,3))
            ),            
        ),
        (
            ( # stage 1a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 1b
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 2a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 2b
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ), 
            ( # stage 3
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3)),
            )
        ),
        (
            ( # stage 1
                ('U', 16, (3,3), (3,3)),
                ('C', 8, (3,3)),
            ),
            ( # stage 2
                ('U', 16, (3,3), (2,3)),
                ('C', 8, (3,3)),
            ),            
        ),
        {'learnrate':0.01, 'regularization':('L1', 1.0), 'temp-loss':('normal',5.0), 'precip-loss':('qp', 1.0)}
    ),

    (# Downsample to 16x16, upsample to match
        (
            ( # stage 1
                ('C', 16, (3,3)),
                ('D', (2,3))
            ),
            ( # stage 2
                ('C', 16, (3,3)),
                ('D', (3,3))
            ),
            ( # stage 3
                ('C', 16, (3,3)),
                ('D', (2,2))
            ),            
        ),
        (
            ( # stage 1a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 1b
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 2a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 2b
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ), 
        ),
        (
            ( # stage 1
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),
            ( # stage 2
                ('U', 16, (3,3), (3,3)),
                ('C', 8, (3,3)),
            ),
            ( # stage 3
                ('U', 16, (3,3), (2,3)),
                ('C', 8, (3,3)),
            ),            
        ),
        {'learnrate':0.01, 'regularization':('L1', 1.0), 'temp-loss':('normal',5.0), 'precip-loss':('qp', 1.0)}
    ),

    (# Downsample to 8x8, upsample to match
        (
            ( # stage 1
                ('C', 16, (3,3)),
                ('D', (2,3))
            ),
            ( # stage 2
                ('C', 16, (3,3)),
                ('D', (3,3))
            ),
            ( # stage 3
                ('C', 16, (3,3)),
                ('D', (2,2))
            ),
            ( # stage 4
                ('C', 16, (3,3)),
                ('D', (2,2))
            ),            
        ),
        (
            ( # stage 1a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 1b
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 2a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
        ),
        (
            ( # stage 1
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),
            ( # stage 2
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),            
            ( # stage 3
                ('U', 16, (3,3), (3,3)),
                ('C', 8, (3,3)),
            ),
            ( # stage 4
                ('U', 16, (3,3), (2,3)),
                ('C', 8, (3,3)),
            ),            
        ),
        {'learnrate':0.01, 'regularization':('L1', 1.0), 'temp-loss':('normal',5.0), 'precip-loss':('qp', 1.0)}
    ),

    (# Downsample to 4x4, upsample to match
        (
            ( # stage 1
                ('C', 16, (3,3)),
                ('D', (2,3))
            ),
            ( # stage 2
                ('C', 16, (3,3)),
                ('D', (3,3))
            ),
            ( # stage 3
                ('C', 16, (3,3)),
                ('D', (2,2))
            ),
            ( # stage 4
                ('C', 16, (3,3)),
                ('D', (2,2))
            ),
            ( # stage 5
                ('C', 16, (3,3)),
                ('D', (2,2))
            ), 
        ),
        (
            ( # stage 1a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
            ( # stage 1b
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
        ),
        (
            ( # stage 1
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),
            ( # stage 2
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),            
            ( # stage 3
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),            
            ( # stage 4
                ('U', 16, (3,3), (3,3)),
                ('C', 8, (3,3)),
            ),
            ( # stage 5
                ('U', 16, (3,3), (2,3)),
                ('C', 8, (3,3)),
            ),            
        ),
        {'learnrate':0.01, 'regularization':('L1', 1.0), 'temp-loss':('normal',5.0), 'precip-loss':('qp', 1.0)}
    ),

    (# Downsample to 2x2, upsample to match
        (
            ( # stage 1
                ('C', 16, (3,3)),
                ('D', (2,3))
            ),
            ( # stage 2
                ('C', 16, (3,3)),
                ('D', (3,3))
            ),
            ( # stage 3
                ('C', 16, (3,3)),
                ('D', (2,2))
            ),
            ( # stage 4
                ('C', 16, (3,3)),
                ('D', (2,2))
            ),
            ( # stage 5
                ('C', 16, (3,3)),
                ('D', (2,2))
            ),
            ( # stage 6
                ('C', 16, (3,3)),
                ('D', (2,2))
            ), 
        ),
        (
            ( # stage 1a
                ('U', 16, (3,3), (2,2)),
                ('C', 16, (3,3))    
            ),
        ),
        (
            ( # stage 1
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),
            ( # stage 2
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),            
            ( # stage 3
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),            
            ( # stage 4
                ('U', 16, (3,3), (2,2)),
                ('C', 8, (3,3)),
            ),            
            ( # stage 5
                ('U', 16, (3,3), (3,3)),
                ('C', 8, (3,3)),
            ),
            ( # stage 6
                ('U', 16, (3,3), (2,3)),
                ('C', 8, (3,3)),
            ),            
        ),
        {'learnrate':0.01, 'regularization':('L1', 1.0), 'temp-loss':('normal',5.0), 'precip-loss':('qp', 1.0)}
    ),

    ## Because our validation code expects at least one stage in each
    ## branch, we can't easily do the case where we downsample all the
    ## way to 1x1 and trivially join the scalars to that.  (For a
    ## similar reason we don't do a case where we don't do any
    ## downsampling and just upsample the scalar inputs.) 
]


def getconfig(n):
    """Get one of the seed configurations supplied with the package"""
    return _configs[n]

def nconfig():
    """Tell how many seed configurations are supplied."""
    return len(_configs)

