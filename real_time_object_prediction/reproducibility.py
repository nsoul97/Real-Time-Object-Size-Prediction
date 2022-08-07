import torch as th
import numpy as np
import os
import random

rng = {'np_rng': None, 'th_rng': None}
def init_rng(seed: int) -> None:
    random.seed(seed)

    rng['np_rng'] = np.random.default_rng(seed)
    rng['th_rng'] = th.Generator()
    rng['th_rng'].manual_seed(0)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':16:8'
    th.random.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = th.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)