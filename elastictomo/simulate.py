from cloudvolume import CloudVolume
import numpy as np
import tifffile

# data
s = 1200

url = "gs://neuroglancer-janelia-flyem-hemibrain/emdata/raw/jpeg"
vg = CloudVolume(url, mip=0, use_https=True)
x,y,z = vg.shape[:3]
cutout = vg[x//2-s//2:x//2+s//2,y//2-s//2:y//2+s//2,z//2:z//2+31,0]

temimg = np.array(cutout[:,:,:,0])
temimg = np.transpose(temimg,(2,1,0))
temimg = temimg.astype(np.float32)
density = -1/31 * np.log(np.maximum(temimg,np.ones_like(temimg)) / temimg.max())

tifffile.imsave('simulated.tif', density)