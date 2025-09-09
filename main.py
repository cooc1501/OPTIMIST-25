# from src.ral_dataset.dataset import RALDataset
# from src.ral_dataset.util import to_zarr

from src.ral_scarr.engines.NICV import NICV
# from src.ral_scarr.engines.snr import SNR 
from src.ral_scarr.file_handling.trace_handler import TraceHandler as th
from src.ral_scarr.container.container import Container, ContainerOptions
from src.ral_scarr.model_values.ciphertext import CipherText
# from src.ral_scarr.model_values.oracle_model import OracleModel

from src.ral_scarr.new_engines.real_batchwise_operation import RealOperationDask
from src.ral_scarr.file_handling.zarr_handler import DaskZarrHandler
from src.ral_scarr.engines.dask_engine import DaskNICV, DaskOracleAES

from multiprocessing.managers import SharedMemoryManager

import os
from time import perf_counter
from cProfile import Profile
from pstats import SortKey, Stats

from matplotlib import pyplot as plt
import numpy as np
import dask
import dask.array as da
import ray
from ray.util.dask import enable_dask_on_ray

# dask distributed scheduler init
# getting a lot of overhead from serializing data between processes here
if __name__ == "__main__":
    # from dask.distributed import Client, LocalCluster
    # cluster = LocalCluster(name='RAL-SCARR', n_workers=2, threads_per_worker=1, processes=True)
    # dask_client = Client(cluster)
    dask_client = None
    # equivalent to:
    #   cluster = LocalCluster()
    #   client = Client(cluster.scheduler.address)
    # print(f"Dask Dashboard: {dask_client.dashboard_link}")

# utility functions
def local_dataset(name: str) -> str:
    return os.path.join(os.path.expanduser("~"), "Nextcloud", "UNH", "Research", "Projects", "SCA Datasets", name + '.zarr')

# test case functions
def dask_tb_1():
    rng = da.random.default_rng()
    a1 = rng.random((50000, 5000))
    a2 = rng.random((50000, 1))

    with Profile() as p:
        transform = RealOperationDask(a1, a2, batches=10, jobs=4)
        
        print(f"results:")
        Stats(p).strip_dirs().sort_stats(SortKey.TIME).print_stats(5)
    
    print(transform[0:50000:100].shape)

def dask_nicv_tb(chunksize: int = 5000, plot: bool = False):
    dataset_name = "SAM4S 100000 random Traces [50c]"
    dset_handle = DaskZarrHandler(local_dataset(dataset_name), chunksize=chunksize)
    t = dset_handle.get_dask_array(0, 0, "traces")
    k = dset_handle.get_dask_array(0, 0, "key")
    t0 = perf_counter()
    transform = DaskNICV(t, k[:], client=dask_client)  # performs computation when initialized
    # TODO: It might make sense to manually call the calculation of the output, that way
    # 'down stream' operations have the oportunity to look at the attributes of the output array while
    # getting initialized before the output array is realized. 
    dt = perf_counter() - t0
    print(f"Chunksize: {chunksize}\t\tCompleted in {dt:.4f} seconds")
    if plot:
        for i in range(transform.outputs[0].shape[0]):
            plt.plot(np.arange(transform.outputs[0].shape[1]), transform.outputs[0][i])
        plt.show()

def dask_oracle_test():
    dataset_name = "SAM4S 100000 random Traces [50c]"
    dset_handle = DaskZarrHandler(local_dataset(dataset_name), chunksize=5000)
    k = dset_handle.get_dask_array(0, 0, "key")
    ct = dset_handle.get_dask_array(0, 0, "ciphertext")
    oracle = DaskOracleAES(aes_type=128, keys=k, texts=ct, round=1, step='sbox', client=None)
    print(oracle.outputs[0])  # this does not get computed until the compute method is called, which is good
    # print(oracle.outputs[0].compute())
    t0 = perf_counter()
    oracle.outputs[0].compute()
    dt = perf_counter() - t0
    print(f"calculated intermediate values in {dt:.4f} seconds")
    input("Press enter to continue...")

def dask_pipeline_test(plot: bool = False):
    # load dataset
    dataset_name = "SAM4S 100000 random Traces [50c]"
    dset_handle = DaskZarrHandler(local_dataset(dataset_name), chunksize=2000)
    
    # select relevant items
    k = dset_handle.get_dask_array(0, 0, "key")
    ct = dset_handle.get_dask_array(0, 0, "ciphertext")
    t = dset_handle.get_dask_array(0, 0, "traces")

    k.dask.visualize()
    k = k.compute(scheduler='single-threaded', rerun_exceptions_locally=True)  # works...
    ct = ct.compute(scheduler='single-threaded', rerun_exceptions_locally=True)  # works...
    t = t.compute(scheduler='single-threaded', rerun_exceptions_locally=True)  # works...


    # leakage model for the targetted cipher
    oracle = DaskOracleAES(aes_type=128, keys=k, texts=ct, round=1, step='sbox', client=None, lazy=True)  # evaluates lazily
    print(f"initialized aes oracle output: {oracle.result}")
    t0 = perf_counter()
    nicv = DaskNICV(t, oracle.result, client=dask_client)  # performs computation when initialized
    dt = perf_counter() - t0
    print(f"Completed pipeline computation in {dt:.4f} seconds")

    if plot:
        for i in range(nicv.nicv.shape[0]):
            plt.plot(np.arange(nicv.nicv.shape[1]), nicv.nicv[i])
        plt.show()

def dask_leakage_model_comparison(plot: bool = False):
    dataset_name = "SAM4S 100000 random Traces [50c]"
    dset_handle = DaskZarrHandler(local_dataset(dataset_name), chunksize=5000)
    k = dset_handle.get_dask_array(0, 0, "key")
    ct = dset_handle.get_dask_array(0, 0, "ciphertext")
    t = dset_handle.get_dask_array(0, 0, "traces")

    oracle_1 = DaskOracleAES(aes_type=128, keys=k, texts=ct, round=1, step='sbox', client=None)  # evaluates lazily
    oracle_2 = DaskOracleAES(aes_type=128, keys=k, texts=ct, round=2, step='sbox', client=None)  # evaluates lazily
    models = [oracle_1, oracle_2]
    for m_i, leakage_model in enumerate(models):
        t0 = perf_counter()
        nicv = DaskNICV(t, leakage_model.outputs[0], client=dask_client)  # performs computation when initialized
        dt = perf_counter() - t0
        print(f"Completed pipeline computation in {dt:.4f} seconds")

        if plot:
            plt.figure(figsize=(10, 6))
            for i in range(nicv.outputs[0].shape[0]):
                plt.plot(np.arange(nicv.outputs[0].shape[1]), nicv.outputs[0][i], label=f"byte {i}", color=(i/nicv.outputs[0].shape[0], 0, 1-(i/nicv.outputs[0].shape[0])))
            plt.legend(loc='upper right')
            plt.title(f"NICV\nLeakage Model {m_i}")
            plt.xlabel("Time sample")
            plt.ylabel("NICV")
            plt.savefig(f"lm-test-{m_i}.png", dpi=300)
            plt.close()

def scarr_nicv_tb(plot: bool = False):
    dataset_name = "SAM4S 100000 random Traces [50c]"

    # if not os.path.isdir(local_dataset(dataset_name)):
    #     dset = RALDataset.load(local_dataset(dataset_name))
    #     # dset.traces = dset.traces.T
    #     # dset.texts = dset.texts.T
    #     # dset.keys = dset.keys.T
    #     to_zarr(dset)

    handler = th(fileName=local_dataset(dataset_name), batchSize=5000)

    # # engine = NICV(model_value=OracleModel(target_round=1, target_op='sbox'))
    engine = NICV(model_value=CipherText())
    # # engine = SNR(model_value=CipherText())

    container = Container(options=ContainerOptions(engine=engine, handler=handler), model_positions=[i for i in range(16)])

    t0 = perf_counter()
    container.run()
    results = np.squeeze(engine.get_result())
    dt = perf_counter() - t0
    print(f"SCARR: calculated NICV in {dt:.4f} seconds")
    
    plt.figure(figsize=(10, 6))
    if plot:
        for i in range(results.shape[0]):
            plt.plot(np.arange(results.shape[1]), results[i, :], label=f"byte {i}", color=(i/results.shape[0], 0, 1-(i/results.shape[0])))
        plt.legend(loc='upper right')
        plt.title("NICV\nTargetting Ciphertext (Input)")
        plt.xlabel("Time sample")
        plt.ylabel("NICV")
        # plt.show()
        plt.savefig('fig.png', dpi=300)
        # plt.close()

def main():
    # dask_oracle_test()
    dask_pipeline_test(plot=True)

    # for i in [5000]:
    #     dask_nicv_tb(chunksize=i, plot=False)
        # scarr_nicv_tb(plot=True)

    # dask_leakage_model_comparison(plot=True)
        
    return

if __name__ == "__main__":
    main()
    raise SystemExit()