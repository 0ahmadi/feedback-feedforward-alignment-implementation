# Feedback-Feedforward Alignment (FFA)
# Ali Ahmadi

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
BATCH_SIZE = 512
EVALS_PER_EPOCH = 10
NOISE_SIGMA = 0.35
BASE_LR = 0.03
SEED = 0
NUM_WORKERS = min(8, os.cpu_count() or 1)
N_SAMPLES = 8
SHOW_PLOTS = True


@dataclass
class Spec:
    key
    title
    n_classes
    build
    image_size = None
    hidden_dim = 1024
    max_train = None
    max_test = None
    manual_check = None
    lr = None

class FFA(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, n_hidden_layers = 1):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * n_hidden_layers + [n_classes]
        self.dims = dims
        self.n_layers = len(dims) - 1
        self.wf = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(dims[i + 1], dims[i]) * 0.03) for i in range(self.n_layers)])
        self.wb = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(dims[i], dims[i + 1]) * 0.03) for i in range(self.n_layers)])

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
        logits = hs[-1] @ self.wf[L - 1].T
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


def build_imagenet(root, train, transform, download):
    return datasets.ImageNet(root, split="train" if train else "val", transform=transform)


def check_imagenet(root):
    devkit = (root / "ILSVRC2012_devkit_t12.tar.gz").exists() or (root / "meta.bin").exists()
    train = (root / "train").exists() or (root / "ILSVRC2012_img_train.tar").exists()
    val = (root / "val").exists() or (root / "ILSVRC2012_img_val.tar").exists()
    if devkit and train and val:
        return ""
    return (f"missing files in {root}. Download ILSVRC2012_devkit_t12.tar.gz, ILSVRC2012_img_train.tar and "
            f"ILSVRC2012_img_val.tar manually from image-net.org and put them there")


SPECS = [
    Spec("mnist", "MNIST", 10, train_flag(datasets.MNIST)),
    Spec("fashion_mnist", "Fashion-MNIST", 10, train_flag(datasets.FashionMNIST)),
    Spec("kmnist", "KMNIST", 10, train_flag(datasets.KMNIST)),
    Spec("emnist", "EMNIST (balanced)", 47, train_flag(datasets.EMNIST, split="balanced")),
    Spec("usps", "USPS", 10, train_flag(datasets.USPS)),
    Spec("cifar10", "CIFAR-10", 10, train_flag(datasets.CIFAR10)),
    Spec("cifar100", "CIFAR-100", 100, train_flag(datasets.CIFAR100)),
    Spec("svhn", "SVHN", 10, split_flag(datasets.SVHN, "train", "test")),
    Spec("stl10", "STL-10", 10, split_flag(datasets.STL10, "train", "test")),
    Spec("imagenet", "ImageNet (32px, 50k subset)", 1000, build_imagenet,
         image_size=32, hidden_dim=1024, max_train=50_000, max_test=10_000,
         manual_check=check_imagenet),
]
SPEC_BY_KEY = {s.key: s for s in SPECS}
COLORS = {s.key: plt.cm.tab10(i % 10) for i, s in enumerate(SPECS)}

def download_all(specs):
    status = {}
    for spec in specs:
        root = DATA_ROOT / spec.key
        print(f"\n=== {spec.title} ===")

        if spec.manual_check is not None:
            msg = spec.manual_check(root)
            status[spec.key] = "ok (found)" if msg == "" else f"MANUAL DOWNLOAD NEEDED: {msg}"
            continue

        try:
            spec.build(root, True, None, True)
            spec.build(root, False, None, True)
            status[spec.key] = "ok"
        except Exception as e:
            status[spec.key] = f"FAILED: {type(e).__name__}: {e}"


def make_transform(image_size):
    steps = [transforms.Grayscale(num_output_channels=1)]
    if image_size is not None:
        steps += [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    steps += [transforms.ToTensor()]
    return transforms.Compose(steps)


def load_split(spec, train):
    limit = spec.max_train if train else spec.max_test
    name = "train" if train else "test"
    cache = CACHE_ROOT / f"{spec.key}_{name}_{spec.image_size}_{limit}.pt"
    if cache.exists():
        return torch.load(cache)

    ds = spec.build(DATA_ROOT / spec.key, train, make_transform(spec.image_size), True)
    if limit is not None and limit < len(ds):
        g = torch.Generator().manual_seed(SEED)
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[:limit].tolist())

    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=NUM_WORKERS)
    xs, ys, hw = [], [], None
    for x, y in tqdm(loader, desc=f"Loading {spec.key} {name}", leave=False):
        hw = list(x.shape[-2:])
        xs.append((x.reshape(x.size(0), -1) * 255).round().to(torch.uint8))
        ys.append(torch.as_tensor(y))
    out = {"x": torch.cat(xs), "y": torch.cat(ys).long(), "hw": hw}

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out

@torch.no_grad()
def evaluate(model, X, y, n_classes, chunk = 4096):
    n = X.shape[0]
    correct, corr_sum, cls_sum, rec_sum = 0, 0.0, 0.0, 0.0

    for i in range(0, n, chunk):
        x, yb = X[i:i + chunk], y[i:i + chunk]
        logits, _ = model.forward_path(x)
        xhat, _ = model.feedback_path(logits)
        target = F.one_hot(yb, n_classes).to(x.dtype)

        correct += (logits.argmax(1) == yb).sum().item()
        cls_sum += ((target - logits) ** 2).sum().item()
        rec_sum += ((x - xhat) ** 2).mean(dim=1).sum().item()

        xc = x - x.mean(dim=1, keepdim=True)
        xh = xhat - xhat.mean(dim=1, keepdim=True)
        corr = (xc * xh).sum(1) / (xc.norm(dim=1) * xh.norm(dim=1) + 1e-8)
        corr_sum += corr.sum().item()

    return {"acc": 100.0 * correct / n, "corr": corr_sum / n, "cls_err": cls_sum / n, "rec_err": rec_sum / n}


@torch.no_grad()
def noisy_accuracy(model, X, y, sigma, chunk = 4096):
    correct = 0
    for i in range(0, X.shape[0], chunk):
        x = (X[i:i + chunk] + sigma * torch.randn_like(X[i:i + chunk])).clamp(0, 1)
        logits, _ = model.forward_path(x)
        correct += (logits.argmax(1) == y[i:i + chunk]).sum().item()
    return 100.0 * correct / X.shape[0]


def auto_lr(spec, input_dim):
    if spec.lr is not None:
        return spec.lr
    return LR_SCALE * BASE_LR * (784 * 256) / (input_dim * spec.hidden_dim)

def train_dataset(spec, device, epochs, batch_size, n_hidden):
    tr, te = load_split(spec, True), load_split(spec, False)
    Xtr, ytr = tr["x"].to(device).float().div_(255), tr["y"].to(device)
    Xte, yte = te["x"].to(device).float().div_(255), te["y"].to(device)

    input_dim = Xtr.shape[1]
    h, w = tr["hw"]
    if int(ytr.max()) >= spec.n_classes or int(yte.max()) >= spec.n_classes:
        raise ValueError(f"{spec.key}: found a label >= n_classes ({spec.n_classes})")

    lr = auto_lr(spec, input_dim)
    print(f"\n{spec.title}: input {h}x{w} = {input_dim}, hidden {n_hidden} x {spec.hidden_dim}, classes {spec.n_classes}, train {len(ytr)}, test {len(yte)}, lr {lr:.4g}")

    torch.manual_seed(SEED)
    model = FFA(input_dim, spec.hidden_dim, spec.n_classes, n_hidden).to(device)

    history = {"epoch": [], "acc": [], "corr": [], "cls_err": [], "rec_err": []}

    def log(epoch_float):
        m = evaluate(model, Xte, yte, spec.n_classes)
        if not all(np.isfinite(v) for v in m.values()):
            return False
        history["epoch"].append(epoch_float)
        for k, v in m.items():
            history[k].append(v)
        return True

    log(0.0)
    n = Xtr.shape[0]
    batches_per_epoch = (n + batch_size - 1) // batch_size
    eval_every = max(1, batches_per_epoch // EVALS_PER_EPOCH)
    diverged, step, t0 = False, 0, time.time()

    pbar = tqdm(range(1, epochs + 1), desc=spec.title)
    for epoch in pbar:
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            model.ffa_step(Xtr[idx], ytr[idx], lr=lr)
            step += 1
            if step % eval_every == 0 or step % batches_per_epoch == 0:
                if not log(step / batches_per_epoch):
                    diverged = True
                    break
        if diverged:
            print(f"  !! {spec.title} diverged (NaN/inf) at epoch {step / batches_per_epoch:.2f}; "
                  f"try a smaller lr in its Spec")
            break
        pbar.set_postfix(acc=f"{history['acc'][-1]:.1f}%", corr=f"{history['corr'][-1]:.3f}")

    result = {
        "title": spec.title, "input_dim": input_dim, "hw": [h, w], "n_classes": spec.n_classes,
        "hidden_dim": spec.hidden_dim, "n_hidden_layers": n_hidden, "lr": lr, "n_train": int(n), "n_test": int(yte.shape[0]),
        "epochs": epochs, "history": history, "diverged": diverged, "seconds": time.time() - t0,
        "clean_acc": None if diverged else history["acc"][-1],
        "noisy_acc": None if diverged else noisy_accuracy(model, Xte, yte, NOISE_SIGMA),
    }

    samples = None
    if not diverged:
        with torch.no_grad():
            x = Xte[:N_SAMPLES]
            xhat, _ = model.feedback_path(model.forward_path(x)[0])
        samples = {
            "orig": x.reshape(-1, h, w).cpu().numpy().astype(np.float16),
            "recon": xhat.clamp(0, 1).reshape(-1, h, w).cpu().numpy().astype(np.float16),
        }
    return result, samples

def load_results():
    results, samples = {}, {}
    if (OUT_ROOT / "results.json").exists():
        results = json.loads((OUT_ROOT / "results.json").read_text())
    if (OUT_ROOT / "samples.npz").exists():
        with np.load(OUT_ROOT / "samples.npz") as z:
            samples = {k: z[k] for k in z.files}
    return results, samples


def save_results(results, samples):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(OUT_ROOT / "samples.npz", **samples)

def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
    })


def plot_curves(results, save_prefix = "ffa_datasets"):
    set_style()
    keys = [s.key for s in SPECS if s.key in results]
    if not keys:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), dpi=200)
    ax_acc, ax_corr, ax_norm, ax_cls, ax_rec, ax_bar = axes.ravel()

    for k in keys:
        r, c = results[k], COLORS[k]
        hst = r["history"]
        chance = 100.0 / r["n_classes"]
        norm_acc = [(a - chance) / (100.0 - chance) for a in hst["acc"]]

        ax_acc.plot(hst["epoch"], hst["acc"], color=c, lw=1.5)
        ax_corr.plot(hst["epoch"], hst["corr"], color=c, lw=1.5)
        ax_norm.plot(hst["epoch"], norm_acc, color=c, lw=1.5)
        ax_cls.plot(hst["epoch"], hst["cls_err"], color=c, lw=1.5)
        ax_rec.plot(hst["epoch"], hst["rec_err"], color=c, lw=1.5)

    ax_acc.set_title("A  Test accuracy", loc="left")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 100)

    ax_corr.set_title("B  Reconstruction correlation", loc="left")
    ax_corr.set_ylabel("Correlation (x, x̂)")
    ax_corr.set_ylim(0, 1)

    ax_norm.set_title("C  Accuracy above chance", loc="left")
    ax_norm.set_ylabel("Normalized accuracy")
    ax_norm.set_ylim(0, 1)

    ax_cls.set_title("D  Classification error", loc="left")
    ax_cls.set_ylabel("‖onehot − output‖² per image")

    ax_rec.set_title("E  Reconstruction error", loc="left")
    ax_rec.set_ylabel("MSE per pixel")

    for ax in (ax_acc, ax_corr, ax_norm, ax_cls, ax_rec):
        ax.set_xlabel("Epochs")
        ax.margins(x=0.02)

    done = [k for k in keys if results[k]["clean_acc"] is not None]
    ypos = np.arange(len(done))[::-1]
    for y, k in zip(ypos, done):
        ax_bar.barh(y + 0.19, results[k]["clean_acc"], height=0.36, color=COLORS[k])
        ax_bar.barh(y - 0.19, results[k]["noisy_acc"], height=0.36, color=COLORS[k], alpha=0.4)
    ax_bar.set_yticks([])
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Test accuracy (%)")
    ax_bar.set_title("F  Noise robustness", loc="left")
    ax_bar.legend(handles=[Line2D([0], [0], color="0.25", lw=6, label="clean"),
                           Line2D([0], [0], color="0.25", lw=6, alpha=0.4, label=f"noisy (σ={NOISE_SIGMA})")],
                  frameon=False, loc="lower right")

    handles = [Line2D([0], [0], color=COLORS[k], lw=2.2) for k in keys]
    labels = [f"{results[k]['title']}  ({results[k]['input_dim']}→{results[k]['hidden_dim']}×{results[k].get('n_hidden_layers', 1)}→{results[k]['n_classes']})"
              for k in keys]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1), h_pad=2.5, w_pad=2.0)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_ROOT / f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_ROOT / f"{save_prefix}.pdf", bbox_inches="tight")


def plot_reconstructions(results, samples, save_prefix = "ffa_reconstructions"):
    set_style()
    keys = [s.key for s in SPECS if f"{s.key}_orig" in samples and s.key in results]
    if not keys:
        return

    fig, axes = plt.subplots(len(keys), 2, figsize=(7.6, 0.95 * len(keys) + 0.7), dpi=200, squeeze=False)
    for row, k in enumerate(keys):
        strips = [np.concatenate(list(samples[f"{k}_{kind}"].astype(np.float32)), axis=1)
                  for kind in ("orig", "recon")]
        for col, strip in enumerate(strips):
            ax = axes[row, col]
            ax.imshow(strip, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[row, 0].set_ylabel(results[k]["title"], rotation=0, ha="right", va="center", fontsize=7.5)

    axes[0, 0].set_title("Original", fontsize=8)
    axes[0, 1].set_title("FFA reconstruction (from the predicted class code)", fontsize=8)
    fig.tight_layout()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_ROOT / f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_ROOT / f"{save_prefix}.pdf", bbox_inches="tight")


def main():
    specs = SPECS
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    n_hidden = N_HIDDEN_LAYERS

    results, samples = load_results()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    for spec in specs:
        try:
            res, smp = train_dataset(spec, device, epochs, batch_size, n_hidden)
        except Exception as e:
            print(f"\nSkipping {spec.title}: {type(e).__name__}: {e}")
            continue

        results[spec.key] = res
        if smp is not None:
            samples[f"{spec.key}_orig"] = smp["orig"]
            samples[f"{spec.key}_recon"] = smp["recon"]

        save_results(results, samples)

    plot_curves(results)
    plot_reconstructions(results, samples)

    print("\n--- final test results ---")
    for s in SPECS:
        if s.key in results and results[s.key]["clean_acc"] is not None:
            r = results[s.key]
            print(
                f"{r['title']:32s} acc {r['clean_acc']:6.2f}%   "
                f"noisy acc {r['noisy_acc']:6.2f}%   "
                f"corr {r['history']['corr'][-1]:.3f}"
            )

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
