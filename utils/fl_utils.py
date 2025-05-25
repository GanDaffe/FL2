import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import OrderedDict
from typing import List
from sklearn.model_selection import train_test_split
import gc
import copy
import matplotlib.pyplot as plt
import pandas as a
from torch import nn

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)

def train(net,
          trainloader,
          criterion,
          optimizer,
          device,
          proximal_mu: float = None):
    net.to(device)
    net.train()
    running_loss, running_corrects, tot = 0.0, 0, 0

    global_params = copy.deepcopy(net).parameters()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)

        loss = criterion(outputs, labels)

        if proximal_mu is not None:
            proximal_term = sum((local_weights - global_weights).norm(2)
                                for local_weights, global_weights in zip(net.parameters(), global_params))
            loss += (proximal_mu / 2) * proximal_term

        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=1)
        tot += images.shape[0]

        running_corrects += torch.sum(predicted == labels).item()
        running_loss += loss.item() * images.shape[0]

        del images, labels, outputs, loss, predicted

    running_loss /= tot
    accuracy = running_corrects / tot

    del global_params, tot
    torch.cuda.empty_cache()
    gc.collect()

    return running_loss, accuracy


def test(net, testloader, device):
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    corrects, total_loss, tot = 0, 0.0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            predicted = torch.argmax(outputs, dim=1)
            corrects += torch.sum(predicted == labels).item()
            total_loss += loss.item() * images.shape[0]
            tot += images.shape[0]

            del images, labels, outputs, predicted

    total_loss /= tot
    accuracy = corrects / tot

    del tot
    torch.cuda.empty_cache()
    gc.collect()

    return total_loss, accuracy

