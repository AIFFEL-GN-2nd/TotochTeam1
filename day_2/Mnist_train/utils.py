import config
import torch.nn.functional as F
import torch.nn as nn
import torch

def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    cumu_acc = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    for _, (data, target) in enumerate(loader):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
        optimizer.zero_grad()

        loss = criterion(network(data), target)
        cumu_loss += loss.item()
        _, predicted = torch.max(network(data).data, 1)
        total += target.size(0)
        cumu_acc += (predicted == target).sum().item()

        loss.backward()
        optimizer.step()
        network.eval() 
    return cumu_loss / len(loader), 100 * cumu_acc / total