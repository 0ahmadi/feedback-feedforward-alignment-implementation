# Feedback-Feedforward Alignment (FFA)
# Ali Ahmadi


import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

DATA_ROOT = Path("./data")
CACHE_ROOT = Path("./cache")
OUT_ROOT = Path("./results")

EPOCHS = 100
N_HIDDEN_LAYERS = 2
LR_SCALE = 1.0
BATCH_SIZE = 512    # 128 ?
EVALS_PER_EPOCH = 10
NOISE_SIGMA = 0.35
BASE_LR = 0.03
SEED = 0
NUM_WORKERS = min(8, os.cpu_count() or 1)
N_SAMPLES = 8


@dataclass
class Spec:
  key
  title
  n_classes
  build
  image_size
  hidden_dim
  max_train
  max_test
  manual_check
  lr


class FFA(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, n_classes, n_hidden_layers = 1):
    super().__init__()
    dims = [input_dim] + [hidden_dim] * n_hidden_layers + [n_classes]
    self.dims = dims

    self.n_layers = len(dims) - 1
    self.wf = torch.nn.ParameterList(
      [torch.nn.Parameter(torch.randn(dims[i+1], dims[i]) * 0.03) for i in range(self.n_layers)]
    )

    self.wb = torch.nn.ParameterList(
      [torch.nn.Parameter(torch.randn(dims[i], dims[i+1]) * 0.03) for i in range(self.n_layers)]
    )

  def forward_path(self, x):
    hs, h = [], x
    for i in range(self.n_layers - 1):
      h = torch.relu(h @ self.wf[i].T)
      hs.append(h)

    logits = h @ self.wf[-1].T
    return logits, hs

  def feedback_path(self, class_code):
    acts, a = [class_code], class_code
    for i in range(self.n_layers - 1, 0, -1):
      a = torch.relu(a @ self.wb[i].T)
      acts.append(a)

    xhat = a @ self.wb[0].T
    return xhat, acts

  @torch.no_grad()
  def ffa_step(self, x, labels, lr = 0.03, recon_weight = 1):
    L, batch = self.n_layers, x.shape[0]
    y = F.one_hot(labels, num_classes=self.dims[-1]).to(x.dtype)

    hs = [x]
    for i in range(L - 1):
      hs.append(torch.relu(hs[-1] @ self.wf[i].T))
    logits = hs[-1] @ self.wf[L-1].T

    deltas = [None] * (L + 1)
    deltas[L] = y - logits
    for l in range(L, 1, -1):
      deltas[l - 1] = (deltas[l] @ self.wb[l - 1].T) * (hs[l - 1] > 0)
    
    dwf = [deltas[l].T @ hs[l - 1] / batch for l in range(1, L + 1)]
    class_code = logits.detach()
    acts = [None] * (L + 1)
    acts[L] = class_code

    for l in range(L, 1, -1):
      acts[l - 1] = torch.relu(acts[l] @ self.wb[l - 1].T)
    xhat = acts[1] @ self.wb[0].T
    e_rec = x - xhat
    eps = [None] * L
    eps[0] = e_rec
    for l in range(1, L):
      eps[l] = (eps[l - 1] @ self.wf[l - 1].T) * (acts[l] > 0)

    dwb = [eps[l - 1].T @ acts[l] / batch for l in range(1, L + 1)]

    for i in range(L):
      self.wf[i].add_(lr * dwf[i])
      self.wb[i].add_(lr * recon_weight * dwb[i])

    return logits, xhat
      




def train_flag(cls, **kwargs):
  def build(root, train, transform, download):
    return cls(root, train=train, transform=transform, download=download, **kwargs)
  return build


def split_flag(cls, train_split, test_split):
  def build(root, train, transform, download):
    return cls(root, split=train_split if train else test_split, transform=transform, download=download)
    return build  
