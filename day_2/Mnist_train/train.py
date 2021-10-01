from dataset import SweepDataset
from model import ConvNet
from optimize import build_optimizer
from utils import train_epoch

import config

def train():
    loader = SweepDataset(config.BATCH_SIZE, config.train_transform)
    model = ConvNet(config.FC_LAYER_SIZE, config.DROOUT).to(config.DEVICE)
    optimizer = build_optimizer(model, config.OPTIMIZER, config.LR)

    for epoch in range(config.EPOCH):
        avg_loss, avg_acc = train_epoch(model, loader, optimizer)
        print(f"TRAIN: EPOCH {epoch + 1:04d} / {config.EPOCH:04d} | Epoch LOSS {avg_loss:.4f} | Epoch ACC {avg_acc:.2f} ")

if __name__ == "__main__":
    train()

