import torch
import os
import matplotlib.pyplot as plt

def save_model(model, path, name):
    os.makedirs(path, exist_ok = True) 
    torch.save(model, os.path.join(path, name))

def save_plot(fig, name, folder=None):
    path = "figures"
    if folder is not None:
        path = os.path.join(path, folder)

    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, f"{name}.png"), bbox_inches='tight')

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)