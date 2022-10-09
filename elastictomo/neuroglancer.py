""" Upload files to neuroglancer
Usage:
python neuroglancer.py volume.tif series1/raw_tilts --resolution 1.0 1.0 1.0
python neuroglancer.py [path/to/vol.tif] [path/to/cloudvolumedir] --resolution 1.0 1.0 1.0
"""

import os
import argparse
from cloudvolume import CloudVolume

import tifffile
import numpy as np

GCLOUD_DIR = 'gs://neuroglancer/kluther/tomography/'

def upload_volume(vol: '(D,H,W) uint8', name: 'str', resolution=(1,1,1), offset=(0,0,0)):
    # checks
    assert(vol.dtype == 'uint8')
    assert(len(vol.shape) == 3)
    
    # transform
    vol = vol.transpose(2,1,0)
    
    # create info file
    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'image',
        data_type       = 'uint8', # Channel images might be 'uint8'
        # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso
        encoding        = 'raw', 
        resolution      = resolution, # Voxel scaling, units are in nanometers
        voxel_offset    = offset, # x,y,z offset in voxels from the origin

        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size      = [ 512, 512, 16 ], # units are voxels
        volume_size     = vol.shape, # e.g. a cubic millimeter dataset
    )

    # create path
    path = os.path.join(GCLOUD_DIR, name)
    
    # print
    print(f'uploading {vol.shape} volume to {path} with resolution={resolution}nm, offset={offset}voxels...')

    # upload
    cvol = CloudVolume(path, info=info, progress=True)
    cvol.commit_info()
    cvol[:,:,:] = vol

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Upload via CloudVolume for visualization in Neuroglancer')
    parser.add_argument('volume', type=str, help='filename of volume to upload')
    parser.add_argument('name', type=str, help='name and location to store in gcloud')
    parser.add_argument('--resolution', type=float, nargs=3, default = (1.0, 1.0, 1.0), help='W H D size in nm of voxels')
    parser.add_argument('--offset', type=int, nargs=3, default = (0,0,0), help='W H D offset of origin in units of voxels')

    args = parser.parse_args()
    
    print('reading file...')
    vol = tifffile.imread(args.volume)
    upload_volume(vol, args.name, args.resolution, args.offset)