import os
import numpy as np
import torch
from algorithm.import_lib import *

def serialize_c(tensor_list: List[torch.Tensor]) -> bytes:
    flat = np.concatenate([t.cpu().detach().numpy().flatten() for t in tensor_list])
    return flat.astype(np.float64).tobytes()

def deserialize_c(byte_data: bytes, model: torch.nn.Module) -> List[torch.Tensor]:
    flat_array = np.frombuffer(byte_data, dtype=np.float64)
    shapes = [p.shape for p in model.parameters()]
    sizes = [torch.prod(torch.tensor(s)).item() for s in shapes]

    tensors = []
    pointer = 0
    for size, shape in zip(sizes, shapes):
        chunk = flat_array[pointer:pointer+size]
        tensor = torch.tensor(chunk, dtype=torch.float32).reshape(shape)
        tensors.append(tensor)
        pointer += size
    return tensors

def set_c_local(partition_id: int, c_local: List[torch.Tensor]):
    path = f"c_local_folder/{partition_id}.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c_local_bytes = serialize_c(c_local)
    with open(path, 'wb') as f:
        f.write(c_local_bytes)

def load_c_local(partition_id: int, model: torch.nn.Module) -> Optional[List[torch.Tensor]]:
    path = f"c_local_folder/{partition_id}.txt"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            c_local_bytes = f.read()
        return deserialize_c(c_local_bytes, model)
    else:
        return None
