"""
Causality tests for the causal convolutional classes.

Causal convolution, applied incrementally, needs to have the same output as when applied to the entire sequence at once.
"""

import torch
import pytest
from source.tcn import CausalConv1d


@pytest.fixture(params=[True, False], name="separable_arg")
def separable_arg(request):
    return request.param


@pytest.fixture
def convolution_fixture(request, separable_arg):
    in_channels = 3
    out_channels = 5
    kernel_size = 3
    dilation = 2
    sequence_length = 10
    input_sequence = torch.randn(1, in_channels, sequence_length)

    convolution_class = request.param

    if convolution_class == CausalConv1d:
        return (
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            input_sequence,
        )


@pytest.mark.parametrize("convolution_fixture", [CausalConv1d], indirect=True)
def test_causal_conv(convolution_fixture):
    convolution, input_sequence = convolution_fixture
    output_causal = convolution(input_sequence)

    sequential_out = []
    for t in range(1, input_sequence.size(2) + 1):
        input_window = input_sequence[:, :, :t]
        sequential_out.append(convolution(input_window)[..., -1:])

    assert torch.allclose(torch.cat(sequential_out, -1), output_causal)
