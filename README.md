
## Import data from the current directory and write to default dump file


```python
from dnnclim import data
import os
import pickle
```


```python
climdata = data.process_dir('.')
```

    tempfiles :  ['./tas_Amon_CESM1-CAM5_rcp26_r1i1p1_200601-210012.nc', './tas_Amon_CESM1-CAM5_rcp26_r2i1p1_200601-210012.nc', './tas_Amon_CESM1-CAM5_rcp26_r3i1p1_200601-210012.nc']
    prfiles :  ['./pr_Amon_CESM1-CAM5_rcp26_r1i1p1_200601-210012.nc', './pr_Amon_CESM1-CAM5_rcp26_r2i1p1_200601-210012.nc', './pr_Amon_CESM1-CAM5_rcp26_r3i1p1_200601-210012.nc']
    topofiles :  ['./sftlf_fx_CESM1-CAM5_historical_r0i0p0.nc', './orog_fx_CESM1-CAM5_historical_r0i0p0.nc']
    Data checks passed.
    dim fields = (95, 192, 288)
    nscen = 3
    dim allfields = (285, 192, 288, 2)
    dim allgmeans = (285, 2)
    total cases: 285
    training cases: 142
      [107  94 227  42  23 222 118 179 155  95 169  66  44 261 128 172 230  29
     255 170 266 218 146 121 152 196 280 101 225  36 174 145 197 154  84 248
      31  20 178 257  28 233 241 144 126 213 122 164  78 278 156 148  12  79
     246   5 210 176 244 260 268  57 160 264 236 279 234 150 113 127 104 238
     185  35  13 192 195 136 202 263 151 206  82  22 284 180  46  85  80  15
     182  88  45  30 103 208 116 194 186 105 231  96  10  74 211   3  19 190
     221 243  39 162 112 271 215 115 100 259   0 120  24  93  73 153 167  90
     102  17 108  40  72 184 191 256 237 161 189  21  63 138 168 193]
    dev cases: 71
      [ 76 253 228   8 129 139  99 181 159 275  49 283  43 204  65   7 188  98
     199 262  77 149 220  32 272 111  41 157 270  68  51 245  61  27 106 212
     258 232  64   1  86  53 130 251 131  33 110 165  50 147 274 200  34 137
      91 219 171  55  52 240 163 209 267 142 117  67 217 214 229  75  58]
    test cases: 72
      [254  70  48 235 265  38  89   9 281 198 239  62  54 216 125  47 175 173
     242 158  25 282 223 249 119 141 132  97  59 207   6 247  69  11 143   2
     135 201  18  71 140 134 109 114 269  14 226 224 183  26 276 166  16 203
     123 133   4 277  56 273 177 250 124  87 205  81 187 252  60  92  37  83]
    topo dim:  (192, 288, 2)


### Print field sizes


```python
for dset in climdata:
    print('dataset: {}'.format(dset))
    if dset == 'topo':
        print('\ttopo: ', climdata[dset].shape)
    else:
        print('\tfld: ', climdata[dset]['fld'].shape)
        print('\tgmean: ', climdata[dset]['gmean'].shape)
```

    dataset: train
    	fld:  (142, 192, 288, 2)
    	gmean:  (142, 2)
    dataset: dev
    	fld:  (71, 192, 288, 2)
    	gmean:  (71, 2)
    dataset: test
    	fld:  (72, 192, 288, 2)
    	gmean:  (72, 2)
    dataset: topo
    	topo:  (192, 288, 2)


### Verify data written to dump file


```python
infile = open('dnnclim.dat','rb')
climdata2 = pickle.load(infile)
infile.close()
```


```python
climdata['train'].keys()
```

    dict_keys(['fld', 'gmean'])


```python
(climdata['train']['fld'] == climdata2['train']['fld']).all()
```

    True


```python

```

## Example model run



```python
from dnnclim import model
import pickle
import time
```

### Set up a model specification

This is a very simple network.  It only downsamples the geographical
data once, and it doesn't have many convolutional layers (which are
what gives this "deep" network most of its depth).  It's mainly
intended to show how to set up a calculation.

```python
dsbranch = (
    ( # stage 1
        ('C', 10, (3,3)),
        ('C', 10, (3,3)),
        ('D', (2,3))
    ),
)

sclbranch = (
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
)

usbranch = (
    ( # stage 1
        ('U', 10, (3,3), (2,3)),
        ('C', 2, (3,3))
    ),
)

otherargs = {'learnrate':0.001}

modelspec = [dsbranch, sclbranch, usbranch, otherargs]
```


### Load the data

This data was prepared with the code from the first section.

```python
infile = open('dnnclim.dat','rb')
dnnclim = pickle.load(infile)

```

### Run training iterations

```python
t1 = time.time()
(loss, ckptfile) = model.runmodel(modelspec, dnnclim, epochs=100, savefile='./dnnclimsave')
t2 = time.time()
print('lowest loss value: {}'.format(loss))
print('checkpoint file: {}'.format(ckptfile))
print('run time: {}'.format(t2-t1))
```

    ***layer param: 370
    	nchannel in= 4  nchannel out= 10
    ***layer param: 910
    	nchannel in= 10  nchannel out= 10
    ***layer param: 190
    ***layer param: 910
    ***layer param: 910
    ***layer param: 910
    ds_stage_sizes: [(96, 96, 10), (192, 288, 4)]
    ***layer param: 1810
    	U-bind accounting:  usnchannel= 10  dsnchannel= 10
    ***layer param: 362
    	usnchannel = 20
    Downsampling branch:	1 stages	final size: (96, 96, 10)
          Scalar branch:	3 stages	final size: (96, 96, 10)
      Upsampling branch:	1 stages	final size: (192, 288, 10)
    
    Total free parameters:	6372
    Model checkpoint at epoch= 0, mean error= 34027.83933229004
    Model checkpoint at epoch= 1, mean error= 32627.49295774648
    Model checkpoint at epoch= 2, mean error= 31273.27282211789
    Model checkpoint at epoch= 3, mean error= 29498.57276995305
    Model checkpoint at epoch= 4, mean error= 26957.767344809596
    Model checkpoint at epoch= 5, mean error= 24379.9186228482
    Model checkpoint at epoch= 6, mean error= 22534.854460093895
    Model checkpoint at epoch= 7, mean error= 21225.827856025036
    Model checkpoint at epoch= 8, mean error= 20373.197704747
    Model checkpoint at epoch= 9, mean error= 19423.651538862807
    Model checkpoint at epoch= 10, mean error= 18504.18779342723
	
		(output from iterations 11-89 omitted)
	
    Model checkpoint at epoch= 90, mean error= 340.6464201877934
    Model checkpoint at epoch= 91, mean error= 333.63967136150234
    Model checkpoint at epoch= 92, mean error= 325.9314032342201
    Model checkpoint at epoch= 93, mean error= 320.48474178403757
    Model checkpoint at epoch= 94, mean error= 315.34419014084506
    Model checkpoint at epoch= 95, mean error= 309.09112545644234
    Model checkpoint at epoch= 96, mean error= 303.35009128847156
    Model checkpoint at epoch= 97, mean error= 298.659070161711
    Model checkpoint at epoch= 98, mean error= 293.6671231090245
    Model checkpoint at epoch= 99, mean error= 288.3284428794992
    lowest loss value: 2263964160.0
    checkpoint file: ./dnnclimsave-99
    run time: 1120.96351480484


