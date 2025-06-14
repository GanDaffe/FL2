import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import OrderedDict
from typing import List
import copy
from torch import nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from algorithm.moon.moon_model import ModelMoon

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict)

def train(net, trainloader, criterion, optimizer, device, num_epochs: int = 1, proximal_mu: float = None):
    net.to(device)
    net.train()
    loss_, acc = 0.0, 0

    global_params = copy.deepcopy(net).parameters()

    for _ in range(num_epochs): 
        running_loss, running_corrects, tot = 0.0, 0, 0

        for images, labels in trainloader:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, labels)

            if proximal_mu is not None:
                proximal_term = sum((local - global_).norm(2)
                                    for local, global_ in zip(net.parameters(), global_params))
                loss += (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)

            tot += images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            running_loss += loss.item() * images.size(0)

        running_loss /= tot
        accuracy = running_corrects / tot

        loss_ += running_loss
        acc += accuracy 
    
    loss_ /= num_epochs
    acc /= num_epochs

    return running_loss, accuracy


def test(net, testloader, device):
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    corrects, total_loss, tot = 0, 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            if isinstance(net, ModelMoon): 
                _, _, outputs = net(images)
            else:
                outputs = net(images)

            loss = criterion(outputs, labels)

            if isinstance(net, ModelMoon):
                _, preds = torch.max(outputs.data, 1)
            else:  
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            corrects += torch.sum(preds == labels).item()
            total_loss += loss.item() * images.size(0)
            tot += images.size(0)

    total_loss /= tot
    accuracy = corrects / tot

    cm = confusion_matrix(all_labels, all_preds)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    TP_ = int(TP.sum())
    FP_ = int(FP.sum())
    FN_ = int(FN.sum())
    TN_ = int(TN.sum())

    return total_loss, accuracy, precision, recall, f1, TP_, FP_, FN_, TN_

