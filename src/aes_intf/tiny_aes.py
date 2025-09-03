"""
CFFI module for the instrumented tiny-aes library.
"""

import cffi
import os
import importlib


tiny_aes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tiny-AES'))

# Globals
tiny_aes_lib = os.path.join(tiny_aes_dir, "tinyaes.so")
tiny_aes_hdr = os.path.join(tiny_aes_dir, "aes.h")

# Definitions from tiny-AES
AES_BLOCKLEN = 16
AES_keyExpSize = 176

# Compilation Macros
AES_TYPE = [("AES128", 16), ("AES192", 24), ("AES256", 32)]  # exclusive  (type, key_len)

# Instantiate foreign function interfaces
ffis = {}
for aes_type in AES_TYPE:
    # ff_intf = cffi.FFI()
    ffis[aes_type[0]] = cffi.FFI()

    # Load header file
    with open(tiny_aes_hdr) as hdr:
        ffis[aes_type[0]].cdef(hdr.read() + f"\ntypedef uint8_t aes_key_t[{aes_type[1]}];\n")

    # Load c file
    with open(os.path.join(tiny_aes_dir, 'aes.c')) as cfile:
        ffis[aes_type[0]].set_source(
        module_name=aes_type[0], 
        source=cfile.read(),
        define_macros=[(aes_type[0], 1), ("AES_KEYLEN", aes_type[1])]
    )

    # if the module hasn't been compiled, import fails and the module gets compiled
    try:
        mod = importlib.import_module("." + aes_type[0], 'src.ral_dataset.aes_intf')
        del(mod)
    except ModuleNotFoundError:
        print(f"Module {aes_type[0]} not found, compiling instrumented Tiny-AES shared library")
        ffis[aes_type[0]].compile(tmpdir=os.path.dirname(__file__))
        # print(f"Compiled CFFI for tiny-AES [{aes_type[0]}]")

