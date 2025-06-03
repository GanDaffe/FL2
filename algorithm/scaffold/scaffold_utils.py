import os
import numpy as np
import torch
from algorithm.import_lib import *


def load_c_local(partition_id: int):
    path = "c_local_folder/" + str(partition_id) +".txt"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            c_delta_bytes = f.read()

        array = np.frombuffer(c_delta_bytes, dtype=np.float64)
        return array
    else:
        return None

def set_c_local(partition_id: int, c_local):
    path = "c_local_folder/" + str(partition_id) +".txt"

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    c_local_list = []
    for param in c_local:
        c_local_list += param.flatten().tolist()

    c_local_numpy = np.array(c_local_list, dtype=np.float64)
    c_local_bytes = c_local_numpy.tobytes()

    with open(path, 'wb') as f:
        f.write(c_local_bytes)
