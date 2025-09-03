"""
An experimental class for interfacing with the existing zarr based SCARR dataset format with
dask SCA workflows
"""

import os
from enum import Enum

import zarr
import dask.array as da

def get_zarr_obj(path: str, verbose: bool = False) -> zarr.Array | zarr.Group:
    if not os.path.isdir(os.path.abspath(path)):
        raise ValueError(f"Dataset {os.path.abspath(path)} not found")
    dat = zarr.open(path, mode='r', zarr_format=None)

    if verbose:
        print(f"Loading dataset from {str(dat.store.root)}\n{dat.tree()}")
    
    return dat

"""
Zarr Group (containing LocalStore) --> DaskArray

The old dataset format uses a local store with the format as follows
/                                                 [ root (group)]
└── 0                                             [ tile_x (group)]
    └── 0                                         [ tile_y (group)]
        ├── ciphertext (100000, 16) uint8         [ ct (array) ]
        ├── key (100000, 16) uint8                [ key (array) ]
        ├── plaintext (100000, 16) uint8          [ pt (array) ]
        └── traces (100000, 5000) uint16          [ trace (array) ]

One thing I could do is make a compatability layer to use the old format with the new methods I'm
working on. This would be pretty straight forward.
Another thing I could do is make a slightly more generic dataset wrapper in a similar vain to the
TraceHandler class. I think some very simple metadata (e.g. which array dimension traces are 
oriented along) could be used to reshape the dataset before calculation if necessary.
"""


def encode_tracehandler_group(tile_x: int, tile_y: int) -> str:
    return f"/{tile_x}/{tile_y}"

"""
For use accessing SCARR formatted datasets, not super clean but it works
"""
class DaskZarrHandler:
    def __init__(self, path: str, chunksize: int = None):
        self.zarr_obj = get_zarr_obj(path)
        self.chunksize = chunksize if (chunksize != None) else 'auto'
        # print(f"Array keys: {[i for i in self.zarr_obj.array_keys()]}")
        # print(f"members: {[i for i in self.zarr_obj.members(max_depth=12)]}")
        # self.tiles = [i for i in self.zarr_obj.members(max_depth=1)]
    
    def get_dask_array(self, tile_x: int, tile_y: int, array: str):
        group = encode_tracehandler_group(tile_x, tile_y)
        try:
            return da.from_zarr(str(self.zarr_obj.store_path), component=group+'/'+array, chunks=(5000))
        except AttributeError:
            return da.from_zarr(str(self.zarr_obj.store.path), component=group+'/'+array, chunks=(5000))


    class ArrayNames(Enum):
        CT = 'ciphertext'
        KY = 'key'
        PT = 'plaintext'
        TR = 'traces'


