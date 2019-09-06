#!/usr/bin/env python

import re
import gzip 
import numpy as np
from sys import argv

def loadsmv(fname):
    ## \nSIZE1=1739;\nSIZE2=1748
    if fname.endswith(".gz"):
        with gzip.open(fname, "rb") as f:
            raw = f.read()
        h = raw[0:1024]
        d = raw[1024:]
        f.close()
        sx = int(re.search("SIZE2=\d+",h).group().split("=")[1])
        sy = int(re.search("SIZE1=\d+",h).group().split("=")[1])
        flat_image = np.frombuffer(d, dtype=np.uint16)
        image = np.reshape(flat_image, ((sx,sy)) ).astype(float)[::-1].T
        return image

    f = open(fname,"rb")
    raw = f.read()
    h = raw[0:1024]
    d = raw[1024:]
    f.close()
    sx = int(re.search("SIZE2=\d+",h).group().split("=")[1])
    sy = int(re.search("SIZE1=\d+",h).group().split("=")[1])
    flat_image = np.frombuffer(d, dtype=np.uint16)
    image = np.reshape(flat_image, ((sx,sy)) ).astype(float)[::-1].T
    return image

  
fname1 = argv[1]
fname2 = argv[2]
image1 = loadsmv(fname1)
image2 = loadsmv(fname2)

print "Root Mean Square (RMS) Error = ", np.sqrt( np.mean( (image1-image2)**2 ) )
