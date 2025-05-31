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

def load_c_local(partition_id: int):
    path = "c_local_folder/" + str(partition_id) +".txt"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            c_delta_bytes = f.read()

        array = np.frombuffer(c_delta_bytes, dtype=np.float64)
        return array
    else:
        return None

# Custom function to serialize to bytes and save c_local variable inside a file
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

def test_scaffold(net, testloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss, tot = 0, 0.0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            tot += images.shape[0]

    accuracy = correct / tot
    loss = loss / tot
    return loss, accuracy
