# Temporal Convolutional Networks (Deep-TCN) in PyTorch

This repository provides an implementation of Temporal Convolutional Networks (TCN) [1] in PyTorch, with focus on
flexibility and fine-grained control over architecture parameters.

Additionally, it incorporates separable convolutions
and pooling layers, contributing to the creation of more streamlined and computationally efficient networks.

## Installation
To install both the package and dependencies needed to run the examples using `pip`, run:
```bash
pip install deep-tcn
pip install deep-tcn[examples]
```
Alternatively, having first cloned the repository, you can do theh same using poetry:
```bash
poetry install
poetry install --all extras
```

## Features

- **Causal Convolutions:** Causal convolutions are employed, making the architecture suitable for sequential data.

- **Separable Convolutions:** The implementation includes support for separable convolutions, aiming to reduce the overall number of network parameters.

- **(Channel) Pooling Layers:** Channel pooling layers are integrated to further enhance the efficiency of the network by reducing dimensionality.

- **Flexible Depth Configuration:** Optionally, network depth can be increased by adding nondilated convolutions after dilated convolutional layers.

- **Residual Blocks with Full Preactivation:** Residual blocks are designed following the "full preactivation" design, according to [2]

- **Supported Normalization Layers:**
  - Group Normalization
  - Weight Normalization
  - Batch Normalization


## Usage
Please refer to the scripts under `examples/` as a starting point.


## References
[1] He et al.: Identity Mappings in Deep Residual Networks. ArXiv, 2016. [Link](https://arxiv.org/pdf/1603.05027.pdf)

[2] Lea et al.: Temporal Convolutional Networks: A Unified Approach to Action Segmentation. ArXiv, 2016. [Link](https://arxiv.org/abs/1608.08242)
