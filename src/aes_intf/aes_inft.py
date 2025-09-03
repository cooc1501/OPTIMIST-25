"""
Interface for interacting with tiny-aes lib to find intermediate values
"""

from dataclasses import dataclass
from typing import Any
from multiprocessing import Lock

from . import tiny_aes
from . import AES128, AES192, AES256
import numpy as np

# TODO: This can't hold a reference to the module because then it cannot
# be serialized between processes :(

# Making this work with multiprocessing might be tricky

@dataclass
class AESVars:
    ctx: Any
    key: Any
    state: Any

@dataclass
class AESObjs:
    vars: AESVars
    lib: Any
    lock = Lock()

ops = {'sbox' : 0, 'shift': 1, 'mix': 2, 'add': 3, 'none': -1}


def init_aes(aes_type: int = 128) -> AESObjs:
    def init_intf_vars(module) -> AESVars:
        ctx = module.ffi.new("struct AES_ctx *")
        key = module.ffi.new("aes_key_t")
        state = module.ffi.new("uint8_t [16]")

        return AESVars(ctx, key, state)

    def init_intf_lib(module):
        lib = module.lib
        return lib

    if aes_type == 128:
        module = AES128
    elif aes_type == 192:
        module = AES192
    elif aes_type == 256:
        module = AES256
    else:
        raise ValueError("aes_type must be 128, 192, or 256")
    
    vars = init_intf_vars(module)
    lib = init_intf_lib(module)
    return AESObjs(vars, lib)

def init_aes_ctx(key_in: bytes, text_in: bytes, aes_obj: AESObjs):
    for i in range(len(key_in)):
        aes_obj.vars.key[i] = key_in[i]
    for i in range(len(text_in)):
        aes_obj.vars.state[i] = text_in[i]

    aes_obj.lib.AES_init_ctx(aes_obj.vars.ctx, aes_obj.vars.key)
    
def calculate_aes(aes_obj: AESObjs, key: bytes, text: bytes, target_round: int = -1, op: str='none') -> np.ndarray:
    init_aes_ctx(key_in=key, text_in=text, aes_obj=aes_obj)
    aes_obj.lib.target(aes_obj.vars.ctx, aes_obj.vars.state, target_round, ops[op])
    intermediates = np.asarray(list(aes_obj.vars.state), dtype=np.uint8)
    return intermediates

def _find_int_values(keys: list[bytes] | np.ndarray, texts: list[bytes] | np.ndarray, aes_obj: AESObjs, op: str = 'none', round: int = -1):
    for arg in [keys, texts]:
        if type(arg) != list or type(arg[0]) != bytes:
            try:
                arg = array_to_bytes(arg)
            except:
                raise ValueError("invalid input argument, keys and texts must be a list of bytes objects")
    
    intermediates = np.empty((len(texts), 16), dtype=np.uint8)
    
    try:
        aes_obj.lock.acquire()
        for i, k in enumerate(keys):
            intermediates[i,:] = calculate_aes(aes_obj, k, texts[i], round, op)
    finally:
        aes_obj.lock.release()

    return intermediates

def array_to_bytes(a: np.ndarray) -> list[bytes]:
    if a.ndim > 2 or a.itemsize > 1:
        raise ValueError("a must be a 2d array of 1 byte values")
    
    return [a[idx].tobytes() for idx in range(a.shape[0])]

"""
Since I can't pass the aes module or any cdata between processes, I need to have a global
interface.
"""
global_aes_objs = {length : init_aes(length) for length in [128, 192, 256]}

def find_int_values(keys: list[bytes] | np.ndarray, texts: list[bytes] | np.ndarray, aes_type: int, op: str = 'none', round: int = -1):
    return _find_int_values(keys, texts, global_aes_objs[aes_type], op, round)

class AES_Intf:
    def __init__(self, aes_type: int = 128):
        self.ops = {'sbox' : 0, 'shift': 1, 'mix': 2, 'add': 3, 'none': -1}

        if aes_type == 128:
            self.init_intf_vars(AES128)
            self.lib = AES128.lib
        elif aes_type == 192:
            self.init_intf_vars(AES192)
            self.lib = AES192.lib
        elif aes_type == 256:
            self.init_intf_vars(AES256)
            self.lib = AES256.lib
        else:
            raise ValueError(f"aes_type ({aes_type}) must be 128, 192, or 256")
    
    def init_intf_vars(self, module):
        self.ctx = module.ffi.new("struct AES_ctx *")
        self.key = module.ffi.new("aes_key_t")
        self.state = module.ffi.new("uint8_t [16]")

    def _init_ctx(self, key_in: bytes, text_in: bytes):
        for i in range(len(key_in)):
            key[i] = key_in[i]
        for i in range(len(text_in)):
            state[i] = text_in[i]
            
        self.lib.AES_init_ctx(self.ctx, self.key)
        
    def _calculate(self, key: list[bytes], text: list[bytes], target_round: int = -1, op: str='none') -> np.ndarray:
        self._init_ctx(key_in=key, text_in=text)
        self.lib.target(self.ctx, self.state, target_round, self.ops[op])
        intermediates = np.asarray(list(state), dtype=np.uint8)
        return intermediates
    
    """
    Calculate intermediate values for tiny-aes 128. Returns AES state for keys and texts after target_round and op
    """
    def find_int_values(self, keys: list[bytes] | np.ndarray, texts: list[bytes] | np.ndarray, op: str = 'none', round: int = -1):
        for arg in [keys, texts]:
            if type(arg) != list or type(arg[0]) != bytes:
                try:
                    arg = self.array_to_bytes(arg)
                except:
                    raise ValueError("invalid input argument, keys and texts must be a list of bytes objects")
        
        intermediates = np.empty((len(texts), 16), dtype=np.uint8)
        
        for i, k in enumerate(keys):
            intermediates[i,:] = self._calculate(k, texts[i], round, op)

        return intermediates
    
    @staticmethod
    def array_to_bytes(a: np.ndarray) -> list[bytes]:
        if a.ndim > 2 or a.itemsize > 1:
            raise ValueError("a must be a 2d array of 1 byte values")
        
        return [a[idx].tobytes() for idx in a.shape[0]]