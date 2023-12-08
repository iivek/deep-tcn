from typing import Union, List, Optional, Iterable

import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal convolutional layer for 1-dimensional input data. Note that this is a somewhat incomplete implementation,
    as it doesn't generalize to strides != 1

    Args:
        in_channels (int):
            Number of input channels.
        out_channels (int):
            Number of output channels.
        kernel_size (int, optional):
            Size of the convolutional kernel. Defaults to 1.
        dilation (int, optional):
            Dilation factor for the convolution. Defaults to 1.
        stride (int, optional):
            Stride of the convolution. Defaults to 1.
        separable (bool, optional):
            Whether to use separate depthwise and pointwise convolutions. Defaults to False.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        dilation=1,
        stride=1,
        separable=False,
    ):
        super(CausalConv1d, self).__init__()
        if separable:
            groups = in_channels
            _out_channels = in_channels
            self.pointwise = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        else:
            groups = 1
            _out_channels = out_channels
            self.pointwise = nn.Identity()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=_out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
            stride=stride,
            groups=groups,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv.forward(x)
        if self.conv.padding[0] > 0:
            x = x[..., : -self.conv.padding[0]]
        x = self.pointwise(x)
        return x


class ChannelPooled(nn.Module):
    """Wrapper module that applies channel pooling, i.e. reduces the number of channels by (weighted) averaging, to the
    output of the provided module.

    Args:
        wrapped_module (nn.Module):
            The module to be wrapped and have channel pooling applied to its output.
        in_channels (int):
            Number of input channels.
        out_channels (int):
            Number of output channels after applying channel pooling.
    """

    def __init__(self, wrapped_module, in_channels, out_channels):
        super().__init__()
        self.inner = wrapped_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_pooling = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x, *args, **kwargs):
        return self.channel_pooling(self.inner(x))


class DropoutWrapper(nn.Module):
    def __init__(self, wrapped, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout1d(dropout)
        self.inner = wrapped

    def forward(self, x):
        return self.dropout(self.inner(x))


class ResElement(nn.Module):
    """An encapsulation of (batch) normalization-activation-convolution-dropout chain, an element of the so-called
    "full pre-activation" residual block. Additionally, support has been provided for the use of channel pooling,
    applied after the nonlinearity.

    Args:
        in_channels (int):
            Number of input channels.
        conv (nn.Module):
            Convolutional module to be encapsulated.
        activation (nn.Module):
            Activation module. If None, an identity activation is used.
        dropout (float, optional):
            Dropout probability. Defaults to 0.0.
        channel_pooling (int, optional):
            Number of output channels after applying channel pooling. Defaults to None.
        group_norm_channels (int, optional):
            Number of channels for group normalization. Defaults to None.
        tags (any, optional):
            Additional information or tags associated with the ResElement. Defaults to None.
        output_channels (int, optional):
            Number of output channels. Defaults to None, in which case it defaults to conv.out_channels.
        weight_norm (bool, optional):
            Whether to use weight normalization. Defaults to False.
    """

    def __init__(
        self,
        in_channels,
        conv: nn.Module,
        activation: nn.Module,
        dropout=0.0,
        channel_pooling: int = None,
        group_norm_channels=None,
        tags=None,
        output_channels=None,
        weight_norm=False,
    ):
        super(ResElement, self).__init__()
        self.in_channels = in_channels
        self.out_channels = (
            output_channels if output_channels else conv.out_channels
        )

        # Group/weight/batch normalization
        if group_norm_channels:
            self.bn = nn.GroupNorm(
                num_groups=group_norm_channels, num_channels=in_channels
            )
        elif weight_norm:
            self.bn = nn.Identity()
            conv.pointwise = nn.utils.weight_norm(conv.pointwise)
            conv.conv = nn.utils.weight_norm(conv.conv)
        else:
            self.bn = nn.BatchNorm1d(in_channels)

        self.conv_and_dropout = DropoutWrapper(conv, dropout)
        if activation:
            self.pooled_activation = activation()
        else:
            self.pooled_activation = nn.Identity()
        if channel_pooling:
            # Wrapping the conv object. Channel pooling is, of course, a causality-preserving operation.
            self.pooled_activation = ChannelPooled(
                self.pooled_activation,
                in_channels=conv.out_channels,
                out_channels=channel_pooling,
            )
        self.tags = tags

    def forward(self, x):
        x = self.bn(x)
        pooled_activation = self.pooled_activation(x)
        x = self.conv_and_dropout(pooled_activation)
        return x, pooled_activation


class ResBlock(nn.Module):
    """Builds the so called "full pre-activation" residual block off the provided ResElement objects.
    Specifically for depth or 2, for example, residual blocks are organised as:

    -+-> BatchNorm -> RELU -> conv -> Dropout -> BatchNorm -> RELU -> conv ->   Dropout -> add ->
     |______________________________________________________________________________________|

    References:
    https://arxiv.org/pdf/1603.05027.pdf

    """

    def __init__(self, res_elements: Iterable[ResElement]):
        super(ResBlock, self).__init__()
        self.elements = nn.ModuleList(res_elements)
        self.adapt = self._adapt(
            self.get_input_channels(), self.get_output_channels()
        )

    @staticmethod
    def _adapt(channels_a, channels_b):
        if channels_a == channels_b:
            adapt = nn.Identity()
        else:
            adapt = nn.Conv1d(
                channels_a, channels_b, kernel_size=1, padding="same"
            )
        return adapt

    def get_input_channels(self):
        return self.elements[0].in_channels

    def get_output_channels(self):
        return self.elements[-1].out_channels

    def forward(self, x, collected_outputs=None):
        """Chain of underlying elements"""
        residual_input = x
        for elem in self.elements:
            x, pre_output = elem(x)
            if (
                collected_outputs is not None
                and elem.tags
                and ("output" in elem.tags)
            ):
                collected_outputs.append(pre_output)
        return x + self.adapt(residual_input)


def chunk_list(l, n):
    n = max(1, n)
    return (l[i : i + n] for i in range(0, len(l), n))


class TcBlock(nn.Module):
    """A block consisting of convolutions with exponentially increasing dilations (together with the accompanying
     architectural elements.)

    Args:
        n (int):
            Number of dilated convolutions in the block. Dilations are exponentially increasing, with basis 2.
        input_channels (int):
            Number of input channels to the block.
        channels (Union[int, Iterable[int]]):
            Number of output channels for each convolution. Can be an integer or an iterable for per-block parameters.
            Passing an iterable as input allows the parameters to be set on a per-convolution basis.
        kernel_size (Union[int, Iterable[int]]):
            Size of the convolutional kernel. Can be an integer or an iterable for per-block parameters. Defaults to 2.
            Passing an iterable as input allows the parameters to be set on a per-convolution basis.
        residual_depth (int):
            The number of convolutional layers in each residual block. Defaults to 2
        dropout (float):
            Inter-layer dropout probability.
        activation (nn.Module):
            Activation function.
        additional_depth (int):
            Number of non-dilated convolutions that will be added after each dilated convolution. Can be used to increase
            network depth without excessively affecting receptive field of the net.
        depth_kernel_size (int):
            Governs kernel_size of the additionally added non-dilated convolutions (see `additional_depth`)
        channel_pooling (Union[int, Iterable[int], None]):
            Channel pooling, via convolution with kernel size of 1. Specifies the number of output channels. For more
            information, see `TcBlock`.
        first_block (bool):
            If True, start with a convolution that stands alone, without being a part of a residual block.
        group_norm_channels (Optional[int]):
            Group normalization. Has precedence over `weight_norm`
        weight_norm (bool):
            Weight normalization. If both `group_norm_channels` and `weight_norm` are None, batch
            normalization will be used, which is the default.
        separable (bool):
            Whether to use separable convolutions. Defaults to False.
    """

    def __init__(
        self,
        n: int,
        input_channels: int,
        channels: Union[int, Iterable[int]],
        kernel_size: Union[int, Iterable[int]] = 2,
        residual_depth: int = 2,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU,
        additional_depth: int = 0,
        depth_kernel_size: int = 2,
        channel_pooling: Union[int, Iterable[int], None] = None,
        first_block: bool = False,
        group_norm_channels: Optional[int] = None,
        weight_norm: bool = False,
        separable: bool = False,
    ):
        super(TcBlock, self).__init__()

        # Expanding the parameters for each architectural element of the model
        conv_kwargs = {
            "kernel_size": [kernel_size] * n
            if isinstance(kernel_size, int)
            else kernel_size,
            "out_channels": [channels] * n
            if isinstance(channels, int)
            else channels,
            "dilation": [2**i for i in range(n)],
            "separable": [separable] * n
            if isinstance(separable, bool)
            else separable,
        }
        if channel_pooling is None or isinstance(channel_pooling, int):
            _channel_pooling = [channel_pooling] * n
        else:
            _channel_pooling = channel_pooling
        block_kwargs = {"channel_pooling": _channel_pooling}
        conv_kwargs = [
            dict(zip(conv_kwargs, t)) for t in zip(*conv_kwargs.values())
        ]
        block_kwargs = [
            dict(zip(block_kwargs, t)) for t in zip(*block_kwargs.values())
        ]

        # Architecture:
        # - convolution with dilation = 1 followed by stacks of dilated convolutions
        # - if `first_block`, the first convolution is not wrapped in a residual block
        res_elements = []
        self.initial_module = nn.Identity()
        _in_channels = input_channels
        for e, (conv, block) in enumerate(zip(conv_kwargs, block_kwargs)):
            if e == 0 and first_block:
                # the first convolution stands alone
                conv["in_channels"] = input_channels
                self.initial_module = DropoutWrapper(
                    CausalConv1d(**conv), dropout
                )
                _in_channels = conv["out_channels"]
                continue

            # Dilated convolution
            res = ResElement(
                conv=CausalConv1d(
                    in_channels=block["channel_pooling"] or _in_channels,
                    **conv
                ),
                activation=activation,
                dropout=dropout,
                group_norm_channels=group_norm_channels,
                weight_norm=weight_norm,
                in_channels=_in_channels,
                **block
            )
            res_elements.append(res)
            _in_channels = res.out_channels

            # Additional nondilated convolutions, for more depth
            conv = {
                "out_channels": conv["out_channels"],
                "kernel_size": depth_kernel_size,
                "separable": conv["separable"],
            }
            nondil_elements = [
                ResElement(
                    conv=CausalConv1d(
                        in_channels=block["channel_pooling"] or _in_channels,
                        **conv
                    ),
                    activation=activation,
                    dropout=dropout,
                    group_norm_channels=group_norm_channels,
                    weight_norm=weight_norm,
                    in_channels=_in_channels,
                    **block
                )
            ] * additional_depth
            res_elements.extend(nondil_elements)
            _in_channels = res.out_channels

        # Creating residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(subset)
                for subset in chunk_list(res_elements, residual_depth)
            ]
        )
        self.receptive_field_size = self._receptive_field_size()

    def _receptive_field_size(self):
        def extract_conv1d_layers(module):
            conv1d_layers = []
            for child in module.children():
                if isinstance(child, nn.Conv1d):
                    conv1d_layers.append(child)
                elif isinstance(child, nn.Module):
                    # Recursively search for Conv1d layers in child modules
                    conv1d_layers.extend(extract_conv1d_layers(child))
            return conv1d_layers

        convs = extract_conv1d_layers(
            self.initial_module
        ) + extract_conv1d_layers(self.res_blocks)
        receptive_field = 1
        for conv in convs:
            kernel_size = conv.kernel_size[0]
            dilation = conv.dilation[0]
            receptive_field += (kernel_size - 1) * dilation
        return receptive_field

    def forward(self, x):
        x = self.initial_module(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        return x


class TcNetwork(nn.Module):
    """Network architecture consisting of stacked TcBlocks.

    Args:
        input_channels (int):
            Number of input channels.
        channels (Union[int, List[List[int]]]):
            Output channels for each dilated convolution. Can be an integer or a nested list.
        kernel_size (Union[int, List[List[int]]]):
            Kernel size for each dilated convolution. Can be an integer or a nested list.
        dilations (Optional[int]):
            Only applicable if `channels` or `kernel_size` are integers; otherwise, inferred from the lists.
        separable (bool):
            Whether to use separable convolutions throughout the entire network.
        blocks (int):
            Number of TCN blocks in the network.
        residual_depth (int):
            Number of convolutional layers in each residual block. Defaults to 2.
        dropout (float):
            Inter-layer dropout probability. For more information, see `TcBlock`.
        input_dropout (float):
            Dropout probability applied to the network inputs.
        additional_depth (int):
            Number of non-dilated convolutions added after each dilated convolution. Can be used to add depth to the
            network without excessively expanding its receptive field. For more information, see `TcBlock`.
        depth_kernel_size (int):
            Kernel size of additionally added non-dilated convolutions (see `additional_depth`).
        activation (nn.Module):
            Activation function. For more information, see `TcBlock`.
        channel_pooling (Optional[int]):
            Channel pooling via convolution with kernel size of 1. Specifies the number of output channels.
            For more information, see `TcBlock`.
        group_norm_channels (Optional[int]):
            Group normalization. Takes precedence over `weight_norm`.
        weight_norm (bool):
            Weight normalization. If both `group_norm_channels` and `weight_norm` are None, batch normalization is used.
        top_pooling (Optional[dict]):
            If specified, additional convolutional layer(s) are added on top. Check examples for more details.
    """

    def __init__(
        self,
        input_channels: int,
        channels: Union[int, List[List[int]]],
        kernel_size: Union[int, List[List[int]]] = 2,
        dilations: Optional[int] = None,
        separable: bool = False,
        blocks: int = 1,
        residual_depth: int = 2,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        additional_depth: int = 0,
        depth_kernel_size: int = 2,
        activation: nn.Module = nn.ReLU,
        channel_pooling: Optional[int] = None,
        group_norm_channels: Optional[int] = None,
        weight_norm: bool = False,
        top_pooling: Optional[dict] = None,
    ):
        super(TcNetwork, self).__init__()

        self.input_channels = input_channels
        self.adapt_inputs = None  # lateral connections

        # The following parameters are either an int or a list of lists. In the latter case they represent per-block
        # parameters.
        self.channels = (
            [[channels] * dilations] * blocks
            if isinstance(channels, int)
            else channels
        )
        self.kernel_size = (
            [[kernel_size] * dilations] * blocks
            if isinstance(kernel_size, int)
            else kernel_size
        )

        block_args = [
            {
                "channels": self.channels[e],
                "kernel_size": self.kernel_size[e],
                "n": dilations,
                "dropout": dropout,
                "residual_depth": residual_depth,
                "activation": activation,
                "channel_pooling": channel_pooling,
                "additional_depth": additional_depth,
                "group_norm_channels": group_norm_channels,
                "weight_norm": weight_norm,
                "depth_kernel_size": depth_kernel_size,
                "separable": separable,
            }
            for e in range(blocks)
        ]
        input_channels, first_block = input_channels, True
        self.blocks = nn.ModuleList()
        for e, blargs in enumerate(block_args):
            block = TcBlock(
                input_channels=input_channels,
                first_block=first_block,
                **blargs
            )
            self.blocks.append(block)
            input_channels, first_block = (
                block.res_blocks[-1].get_output_channels(),
                False,
            )

        self.input_dropout = nn.Dropout(input_dropout)

        # Top convolution(s), optional and customizable.
        out_channels = input_channels
        if top_pooling is None:
            self.top_pooling = nn.Identity()
        else:
            _top_pooling = []
            for kwargs in top_pooling["conv_kwargs"]:
                _top_pooling.append(
                    ResElement(
                        conv=CausalConv1d(in_channels=out_channels, **kwargs),
                        activation=None,
                        dropout=0.0,
                        group_norm_channels=group_norm_channels,
                        weight_norm=weight_norm,
                        in_channels=out_channels,
                        output_channels=kwargs["out_channels"],
                    )
                )
                out_channels = kwargs["out_channels"]
            self.top_pooling = nn.Sequential(
                *[
                    ResBlock(subset)
                    for subset in chunk_list(_top_pooling, residual_depth)
                ]
            )
        self.receptive_field_size = sum(
            [b.receptive_field_size for b in self.blocks]
        )

    def forward(self, inp, *args, **kwargs):
        x = self.input_dropout(inp)
        block_outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            block_outputs.append(x)
        x = self.top_pooling(x)

        return {"output": x, "block_outputs": block_outputs}
