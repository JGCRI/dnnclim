# dnnclim: a deep neural network emulator for Earth System Models


The example code below illustrates how to use the software.  The
example data isn't included, but everything we used here is publicly
available from the CMIP5 archive.

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
    dim allfields = (285, 2, 192, 288)
    dim allgmeans = (285, 2)
    total cases: 285
    training cases: 142
      [ 39  54 234 242 247  44 225  67  25 168 272  40 155 115 153 205 130 198
     188 172  66 154 110 253 138  72  92 119 232 164  84  10  83  71 125  42
     203   0 249  60 178 227  73 162 135 139 102 194 236 166 231   2 146 128
     265 145 240 251  26 143  90  95  61 101   9 129 228  50 212  29 214 120
     186 281 209 246  43 140 114 121 271 142 160  82 159 116 131 275 199  74
     233 237 206  68 134 173 113 264  46 107 248  56 261  85 124 167 182 239
     126 224  18 144 132  36  28 111 191 161 245  75 278  89  31 258  16 147
     181 259   1 190   5 256 171 123 187 260 108  34 244 133 270  78]
    dev cases: 71
      [226  55  91 222  87 267 103 177 165 183  77 257  22 223   6 216 152 158
      69 104  45  62 255 215 241 141 185  51 238  98  48 117 200 189  35  20
     208  99 192 219 137  94  93  88 127 279 175 136  59 179  27 266  17 100
     150   3  86  47 276 235 283 184  58 217  30 213  37 221  76  19  38]
    test cases: 72
      [210 148 196 280 204  97 105 106  63  57 118 211 122 254 163  23 207 243
     263 269 193 157 229 151  21  81 201 250 112 273 284 252  53  65  79 218
     170   8  80 262 268  70 149  52  12  96  15  64   7   4 277 176  41 169
     109 230 202 195  11 282 220  32 180  24 156  49 174 197  33  14  13 274]
    topo dim:  (2, 192, 288)


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
    	fld:  (142, 2, 192, 288)
    	gmean:  (142, 2)
    dataset: dev
    	fld:  (71, 2, 192, 288)
    	gmean:  (71, 2)
    dataset: test
    	fld:  (72, 2, 192, 288)
    	gmean:  (72, 2)
    dataset: topo
    	topo:  (2, 192, 288)


## Verify data written to dump file


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
climdata2['train'].keys()
```




    dict_keys(['fld', 'gmean'])




```python
(climdata['train']['fld'] == climdata2['train']['fld']).all()
```




    True




```python

```
