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
