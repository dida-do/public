import torch
from torch import nn
from xrocket.convolutions import RocketConv
from xrocket.multichannel import ChannelMix
from xrocket.pooling import PPVThresholds

class DilationBlock(nn.Module):
    """MiniRocket block for transformation of timeseries at a single dilation value.

    This layer serves to perform the encoding of an input timeseries with a fixed
    dilation value.
    A DilationBlock consists of the following three sublayers:
     - RocketConv
     - ChannelMix
     - PPVThresholds
    A forward pass transforms a tensor of shape (Batch * Channels * Timeobs)
    into a tensor of shape (Batch * (Features/Dilation)).

    This implementation is based on the descriptions in:
    Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
    "Minirocket: A very fast (almost) deterministic transform for time series classification."
    Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

    The block structure deviates from the original paper and sublayers have differences as
    explained in the respective implementations.

    Attributes:
        in_channels: Number of channels in each timeseries.
        dilation: The dilation value to apply to the convolutional kernels.
        num_thresholds: The number of thresholds per channel combination.
        combination_order: The maximum number of channels to be interacted.
        combination_method: The channel mixing method, either 'additive' or 'multiplicative'.
        kernel_length: Number of paramters in each kernel, default = 9.
        num_kernels: The number of kernels considered in the module.
        num_combinations: The number of channel combinations considered in the module.
        feature_names: (pattern, dilation, channels, threshold) tuples to identify features.
        is_fitted: Indicates that thresholds are fitted to a data example.
    """

    def __init__(
        self,
        in_channels: int,
        dilation: int,
        num_thresholds: int = 1,
        combination_order: int = 1,
        combination_method: str = "additive",
        kernel_length: int = 9,
    ):
        """Set up attributes including quantile values for the layer.

        Args:
            in_channels: Number of channels in each timeseries.
            dilation: The dilation value to apply to the convolutional kernels.
            num_thresholds: The number of thresholds per channel combination.
            combination_order: The maximum number of channels to be interacted.
            combination_method: Keyword for the channel mixing method, default='additive'.
            kernel_length: Number of paramters in each kernel, default = 9.
        """
        super().__init__()

        # set up constituent layers
        self.conv = RocketConv(
            in_channels=in_channels,
            dilation=dilation,
            kernel_length=kernel_length,
        )
        self.mix = ChannelMix(
            in_channels=self.conv.out_channels,
            in_kernels=self.conv.num_kernels,
            order=combination_order,
            method=combination_method,
        )
        self.thresholds = PPVThresholds(
            num_thresholds=num_thresholds,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass to calculate a feature vector.

        Pooling thresholds will be fit to the first example if not yet fitted.

        Args:
            x: Tensor of shape (Batch * Channels * Timeobs)

        Returns:
            x: Tensor of shape (Batch * (Features/Dilations))
        """
        x = self.conv(x)
        x = self.mix(x)
        x = self.thresholds(x)
        x = torch.flatten(x, start_dim=1)
        return x

    @property
    def in_channels(self) -> int:
        """The number of incoming channels."""
        return self.conv.in_channels

    @property
    def dilation(self) -> int:
        """The value to dilute the kernels with over the time dimension."""
        return self.conv.dilation

    @property
    def combination_order(self) -> int:
        """The highest number of channels to combine in a feature."""
        return self.mix.order

    @property
    def num_kernels(self) -> int:
        """The number of kernels in the convolutional block."""
        return self.conv.num_kernels

    @property
    def num_combinations(self) -> int:
        """The total number of channel combinations."""
        return self.mix.num_combinations

    @property
    def num_thresholds(self) -> int:
        """The number of thresholds to apply to each channel combinations."""
        return self.thresholds.num_thresholds

    def fit(self, x: torch.Tensor) -> None:
        """Obtain pooling threshold values from an input.

        Accepts either a single example or a batch as an input.

        Args:
            x: Tensor of shape (Channels * Timeobs) or
                Tensor of shape (Batch * Channels * Timeobs)
        """
        x = self.conv(x)
        x = self.mix(x)
        self.thresholds.fit(x)

    @property
    def is_fitted(self) -> bool:
        """Indicates if module biases were fitted to data."""
        return self.thresholds.is_fitted

    @property
    def feature_names(self) -> list[tuple]:
        """(pattern, dilation, channels, threshold) tuples to identify features."""
        assert self.is_fitted, "module needs to be fitted for thresholds to be named"
        feature_names = [
            (
                str(pattern),
                self.dilation,
                str(channels),
                f"{threshold:.4f}",
            )
            for pattern, channels, threshold in zip(
                self.conv.patterns * self.num_combinations * self.num_thresholds,
                self.mix.combinations * self.num_thresholds,
                self.thresholds.thresholds,
            )
        ]
        return feature_names
