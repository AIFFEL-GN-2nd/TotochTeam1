import math
import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCH = 5
BATCH_SIZE = 128
FC_LAYER_SIZE = 128
WEIGHT_DECAY = 0.0005
LR = 1e-3
DROOUT = 0.5
ACTIVATION = 'relu'
OPTIMIZER = 'adam'

train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

