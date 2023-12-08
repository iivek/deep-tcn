# Temporal Convolutional Networks (Deep-TCN) in PyTorch

This repository provides an implementation of Temporal Convolutional Network (TCN) architectures [1] in PyTorch,
with focus on flexibility and fine-grained control over the network architecture. The design includes support for
separable convolutions and pooling layers for overall computationally more efficient architectures.

## Features

- **Causal Convolutions:** Causal convolutions are employed, making the architecture suitable for sequential data.

- **Separable Convolutions:** The implementation includes support for separable convolutions, aiming to reduce the overall number of network parameters.

- **(Channel) Pooling Layers:** Channel pooling layers are integrated to further enhance the efficiency of the network by reducing dimensionality.

- **Flexible Depth Configuration:** Optionally, network depth can be increased by adding nondilated convolutions after dilated convolutional layers.

- **Residual Blocks with Full Preactivation:** Residual blocks are designed following the "full preactivation" design from [2]

- **Supported Normalization Layers:**
  - Group Normalization
  - Weight Normalization
  - Batch Normalization

## Usage

```python
# Example usage code goes here
```

[1] He et al.: Identity Mappings in Deep Residual Networks. ArXiv, 2016. [Link](https://arxiv.org/pdf/1603.05027.pdf)
[2] Lea et al.: Temporal Convolutional Networks: A Unified Approach to Action Segmentation. ArXiv, 2016. [Link](https://arxiv.org/abs/1608.08242)
