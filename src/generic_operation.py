from time import perf_counter_ns

import dask
import dask.array as da
import dask.graph_manipulation
import numpy as np
import numba as nb
import scalib.config
from scalib.metrics import SNR as scalibSNR
from scalib.tools import ContextExecutor as SCALIBExecutor

from .aes_intf.aes_inft import find_int_values

"""
Base class for all operations
"""

class GenericOperation:
    # passed on init
    inputs: list[np.typing.ArrayLike]

    # allocated after init
    outputs: list[np.typing.ArrayLike]

    # internal variables
    _lazy: bool
    output_attrs: list[str]

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.inputs = []
        obj.outputs = []
        obj.output_attrs = []
        obj._lazy = False if not kwargs.get('lazy') else True

        return obj

    def __init__(self, *args, **kwargs):
        self._begin()

    def _operation(self):
        raise NotImplementedError()
    
    def _validate_input(self):
        try:
            for cond in self.input_conditions:
                if not cond:
                    raise ValueError("input condition not met")
        except AttributeError as e:
            e.args = (f"No input specification for {self.__class__.__name__}: input_conditions must be declared in {self.__class__.__name__}.__new__", *e.args)
            # raise e
            print("Ignoring exception:")
            print(e)

    def _compute_output(self):
        for i, out in enumerate(self.output_attrs):
            if type(self.__getattribute__(out)) == da.Array:
                self.__setattr__(out, self.__getattribute__(out).compute())

    def _begin(self):
        try:
            self._validate_input()
        except NotImplementedError as e:
            print("Ignoring exception:")
            print(e)
        
        self._operation()

        if not self._lazy:
            self._compute_output()
    
    def _set_input(self, array: np.typing.ArrayLike, name: str):
        self.__setattr__(name, array)
        self.inputs.append(self.__getattribute__(name))
    
    def _set_output(self, array: np.typing.ArrayLike, name: str):
        self.__setattr__(name, array)
        self.outputs.append(self.__getattribute__(name))
        self.output_attrs.append(name)


# normalized inter-class variance (NICV) operation (proof of concept)
class NICV(GenericOperation):
    traces: np.typing.ArrayLike  # input
    labels: np.typing.ArrayLike  # input
    nicv: np.typing.ArrayLike  # output

    def __new__(cls, traces, labels, *args, **kwargs):
        # call DaskEngine.new()
        obj = super().__new__(cls, *args, **kwargs)

        # set operation inputs
        obj._set_input(traces, 'traces')
        obj._set_input(labels, 'labels')

        # specify input conditions
        obj.input_conditions = [ 
            obj.labels.ndim in [1, 2], 
            obj.traces.ndim == 2,
            obj.traces.shape[0] == obj.labels.shape[0]
        ]
        
        # declare any internal variables (this does not have to be done in __new__, it could be done in _operation)
        obj.trace_counts = np.zeros((2**(obj.labels.itemsize*8)), dtype=np.uint16)
        obj.sum = np.zeros((2**(obj.labels.itemsize*8), obj.traces.shape[-1]), dtype=np.float32)
        obj.sum_sq = np.zeros((2**(obj.labels.itemsize*8), obj.traces.shape[-1]), dtype=np.float32)

        # return the new instance
        return obj

    def _operation(self):
        # views to reshaped input
        tv = self.traces
        lv = np.atleast_2d(self.labels).reshape(tv.shape[0], self.labels[0].size)
        
        # initialize output
        nicv = da.from_array(np.zeros((lv[0].size, tv.shape[1]), dtype=np.float32))

        lv_unq, cnt_unq = np.unique(lv, return_counts=True)

        sums = da.blockwise(
            self._accumulator_sum_2, 'xklj', tv, 'ij', lv, 'ik', lv_unq, 'l', new_axes={'x': 2},
            dtype=np.float32, concatenate=True, meta=np.array((lv_unq.size, tv.shape[1]))
        )

        for i in range(lv.shape[1]):  
            nicv[i] = da.blockwise(
                self._finalize, 'j', sums[0, i], 'lj', sums[1, i], 'lj', cnt_unq, 'l',
                dtype=np.float32, concatenate=True, meta=np.array((tv.shape[1], )) 
            )
        
        self._set_output(nicv, 'nicv')
    
    """
    Calculate sum and sum of squared value of traces over time w.r.t. trace label
    """
    @staticmethod
    @nb.njit(cache=True)
    def _accumulator_sum(traces: np.ndarray, labels: np.ndarray, unique_labels: np.ndarray):
        sums = np.zeros((2, unique_labels.shape[0], traces.shape[1]), dtype=np.float32)

        for row in nb.prange(traces.shape[0]):
            sums[0, labels[row]] += traces[row, :]
            sums[1, labels[row]] += np.square(traces[row, :])
    
        return sums

    """
    Performs the same operation as _accumulator_sum(), but for multiple dimensions of labels.
    """
    @staticmethod
    @nb.njit(cache=True, parallel=True)
    def _accumulator_sum_2(traces: np.ndarray, labels: np.ndarray, unique_labels: np.ndarray):
        sums = np.zeros((2, labels.shape[1], unique_labels.shape[0], traces.shape[1]), dtype=np.float32)

        for l in nb.prange(labels.shape[1]):
            for row in nb.prange(traces.shape[0]):
                sums[0, l, labels[row, l]] += traces[row, :]
                sums[1, l, labels[row, l]] += np.square(traces[row, :])
        
        return sums

    """
    Calculate final result from accumulator results
    """
    @staticmethod
    @nb.njit(cache=True)
    def _finalize(sums: da.Array, sums_sq: da.Array, counts: np.ndarray):
        mean = np.sum(sums, axis=0) / np.sum(counts)
        signals = (((sums / counts[:, None]) - mean))**2
        signals *= (counts / counts.shape[0])[:, None]
        signals = np.sum(signals, axis=0)

        noises = np.sum(sums_sq, axis=0) / np.sum(counts) - (mean)**2
        return signals / noises

# AES oracle interface wrapped as an operation for inclusion in SCA workflows (proof of concept)
class OracleAES(GenericOperation):
    result: np.typing.ArrayLike

    def __new__(cls, aes_type: int, keys, texts, round: int, step: str, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        
        obj.aes_type = aes_type
        obj._set_input(keys, 'keys')
        obj._set_input(texts, 'texts')
        obj._set_input(round, 'round')
        obj._set_input(step, 'step')

        # specify input conditions
        obj.input_conditions = [ 
            obj.aes_type in [128, 192, 256], 
            obj.keys.shape == obj.texts.shape,
            obj.keys.shape[1] == 16
        ]

        return obj

    def _operation(self):
        result = da.blockwise(
            find_int_values, 'ij', self.keys, 'ij', self.texts, 'ij', dtype=np.uint8, 
            aes_type = self.aes_type, round=self.round, op=self.step, 
            meta=np.array(self.keys.shape)
        )
        self._set_output(result, 'result')

# signal to noise ratio operation (wraps the SCAlib rust implementation of SNR, proof of concept)
class SNR(GenericOperation):
    traces: np.typing.ArrayLike  # input
    labels: np.typing.ArrayLike  # input
    snr: np.typing.ArrayLike  # output

    def __new__(cls, traces, labels, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)

        # set operation inputs
        obj._set_input(traces, 'traces')
        obj._set_input(labels, 'labels')

        # specify input conditions
        obj.input_conditions = [ 
            obj.labels.ndim in [1, 2], 
            obj.traces.ndim == 2,
            obj.traces.shape[0] == obj.labels.shape[0]
        ]

        # return the new instance
        return obj

    def _operation(self):
        tv = self.traces.astype(np.int16)
        lv = np.atleast_2d(self.labels).reshape(tv.shape[0], self.labels[0].size).astype(np.uint16)
        tv = tv.rechunk({0: 'auto', 1: -1})
        
        # calculate unique labels
        lv_unq = np.unique(lv)

        snr_intf = scalibSNR(nc=lv_unq.size)
        start = 0
        ts = []
        for block in tv.blocks:
            ts.append(dask.delayed(snr_intf.fit_u)(block, lv[start:start+block.shape[0]]))
            start += block.shape[0]

        snr = dask.graph_manipulation.bind(dask.delayed(snr_intf.get_snr)(), tuple(ts))

        self._set_output(snr, 'snr')

    # SCAlib implements its own threading, so we compute the result on the main thread
    def _compute_output(self):
        with scalib.config.Config(n_threads=8).activate():
            for i, out in enumerate(self.output_attrs):
                try:
                    # scheduler='sync' executes compute on the main thread
                    self.__setattr__(out, self.__getattribute__(out).compute(scheduler='sync'))
                except Exception as e:
                    print(f"Could not compute output for {__class__.__name__}")
                    print(e)