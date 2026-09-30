# feedback-feedforward-alignment-implementation

This repository is an implementation of the paper:

**[Brain-like Flexible Visual Inference by Harnessing Feedback-Feedforward Alignment](https://pmc.ncbi.nlm.nih.gov/articles/PMC11567678/)**

by Tahereh Toosi and Elias B. Issa.

The original work introduces **Feedback-Feedforward Alignment (FFA)**, a learning algorithm that jointly trains feedforward and feedback pathways using local learning rules, enabling both classification and reconstruction while providing a feedback pathway that can be used for visual inference.

## Results

![FFA Dataset Results](https://github.com/0ahmadi/feedback-feedforward-alignment-implementation/blob/main/files/ffa_datasets.png)

![FFA Reconstructions](https://github.com/0ahmadi/feedback-feedforward-alignment-implementation/blob/main/files/ffa_reconstructions.png)

## Current Implementation

This implementation currently provides:

* A deep FFA architecture with multiple hidden layers.
* Separate feedforward (`wf`) and feedback (`wb`) weights.
* Local weight updates for both pathways.
* Classification through the feedforward pathway.
* Reconstruction through the feedback pathway.
* Feedback-based credit assignment for deeper layers.
* Support for multiple image datasets, including MNIST, Fashion-MNIST, KMNIST, EMNIST, USPS, CIFAR-10, CIFAR-100, SVHN, STL-10, and ImageNet.
* Grayscale preprocessing and optional resizing.
* Training and evaluation metrics including accuracy, reconstruction correlation, classification error, and reconstruction error.
* Gaussian-noise robustness evaluation.
* Automatic result caching and visualization of training curves and reconstructions.

The implementation is intentionally kept simple and uses a static configuration rather than command-line flags.

## Planned Work

The following extensions are planned:

### Convolutional FFA

Extend the current fully connected implementation to convolutional layers while preserving the FFA learning principle and local update rules.

### Transformer-based FFA

Explore how the FFA framework can be adapted to Transformer architectures and attention-based representations.

### Layer Normalization

Add **Layer Normalization between layers without learnable parameters** (no gain or bias).

The goal is to stabilize the scale of activations across layers without introducing additional learned parameters that would require a separate local update rule. This is independent of error clipping and is intended to help prevent unbalanced activation growth or decay in deeper networks.

### Direct Feedback Alignment

Replace the current layer-by-layer error propagation with **Direct Feedback Alignment (DFA)**.

Instead of propagating the classification error sequentially through intermediate layers, each layer would receive the output error directly through its own fixed feedback matrix. This could reduce the degradation of the learning signal in deeper networks.

A possible extension is to combine DFA with the FFA feedback pathway, retaining the decoder-like feedback network while using direct error signals for feedforward weight updates.

## Reference

Toosi, T., & Issa, E. B. (2023). *Brain-like Flexible Visual Inference by Harnessing Feedback-Feedforward Alignment*. Advances in Neural Information Processing Systems (NeurIPS).
