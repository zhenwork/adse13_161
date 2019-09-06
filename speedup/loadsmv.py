import re

def loadsmv(fname):
    ## \nSIZE1=1739;\nSIZE2=1748
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
