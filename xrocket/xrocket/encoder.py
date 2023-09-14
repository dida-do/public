import math
import torch
from torch import nn
from xrocket.block import DilationBlock

class XRocket(nn.Module):
    """Explainable ROCKET module for timeseries embeddings.

    Serves to encode a (multivariate) timeseries into a fixed-length feature vector.
    I.e., a forward pass transforms a tensor of shape (Batch * Channels * Timeobs)
    into a tensor of shape (Batch * Features).
    The implementation is such that the origin of each feature can be traced.

    This implementation is based on the descriptions in:
    Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
    "Minirocket: A very fast (almost) deterministic transform for time series classification."
    Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

    The implemented block structure deviates from the original paper but the calculations are
    almost identical. Please refer to the sublayer implementations for details.

    Attributes:
        in_channels: Number of channels in each timeseries.
        max_kernel_span: Number of time-observations in a typical timeseries.
        combination_order: The maximum number of channels to be combined.
        combination_method: The channel mixing method, either 'additive' or 'multiplicative'.
        feature_dims: The number of values in each feature dimension.
        num_features: The total number of feature embeddings.
        is_fitted: Indicates of the module has been fitted to data.
        feature_names: List of feature name tuples (pattern, dilation, channels, threshold).
    """

    def __init__(
        self,
        in_channels: int,
        max_kernel_span: int,
        combination_order: int = 1,
        combination_method: str = "additive",
        feature_cap: int = 10_000,
        kernel_length: int = 9,
        max_dilations: int = 32,
    ):
        """Set up attributes for all sub-layers.

        Args:
            in_channels: The number of channels in the data.
            max_kernel_span: Number of time-observations in a typical timeseries.
            combination_order: The maximum number of channels to be interacted.
            combination_method: Keyword for the channel mixing method, default='additive'.
            feature_cap: Maximum number of features to be considered.
            kernel_length: The length of the 1D convolutional kernels.
            max_dilations: The maximum number of distinct dilation values.
        """
        super().__init__()
        self.dilations = self._deduce_dilation_values(
            max_kernel_span=max_kernel_span,
            kernel_length=kernel_length,
            max_dilations=max_dilations,
        )
        num_kernels = len(
            torch.combinations(torch.arange(kernel_length), kernel_length // 3)
        )
        num_combinations = sum(
            [
                len(torch.combinations(torch.arange(in_channels), order + 1))
                for order in range(combination_order)
            ]
        )
        num_mix_channels = self.num_dilations * num_kernels * num_combinations
        if feature_cap < num_mix_channels:
            raise ValueError(
                (
                    f"input combinations ({num_mix_channels}) "
                    f"greater than feature cap ({feature_cap})."
                )
            )
        num_thresholds = feature_cap // num_mix_channels

        # set up rocket blocks
        self.blocks = nn.ModuleList()
        for dilation in self.dilations:
            self.blocks.append(
                DilationBlock(
                    in_channels=in_channels,
                    dilation=dilation,
                    num_thresholds=num_thresholds,
                    combination_order=combination_order,
                    combination_method=combination_method,
                )
            )

    def _deduce_dilation_values(
        self,
        max_kernel_span: int,
        kernel_length: int,
        max_dilations: int,
    ) -> None:
        """Create dilation values following the scheme in the original paper.

        Dilation values are chosen according to the number of observations
        with the formula in the paper.
        """
        max_exponent = math.log((max_kernel_span - 1) / (kernel_length - 1), 2)
        integers = (
            (2 ** torch.linspace(0, max_exponent, max_dilations)).to(dtype=int).tolist()
        )
        return list(set(integers))

    @property
    def num_dilations(self) -> int:
        """The number of distinct dilation values for each kernel."""
        return len(self.dilations)

    @property
    def num_kernels(self) -> int:
        """The number of convolutional kernels per dilation."""
        return self.blocks[0].num_kernels

    @property
    def num_combinations(self) -> int:
        """The number of channel combinations per dilation."""
        return self.blocks[0].num_combinations

    @property
    def num_thresholds(self) -> int:
        """The number of pooling thresholds per channel combination per dilation."""
        return self.blocks[0].num_thresholds

    @property
    def feature_dims(self) -> dict:
        """A dictionary with the number of each feature attribute."""
        feature_dims = {
            "num_kernels": self.num_kernels,
            "num_dilations": self.num_dilations,
            "num_combinations": self.num_combinations,
            "num_thresholds": self.num_thresholds,
        }
        return feature_dims

    @property
    def num_features(self) -> int:
        """The total number of feature encodings used by the layer."""
        num_features = (
            self.num_kernels
            * self.num_dilations
            * self.num_combinations
            * self.num_thresholds
        )
        return num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass to calculate timeseries feature encodings.

        Args:
            x: Tensor of shape (Batch * Channels * Timeobs)

        Returns:
            out: Tensor of shape (Batch * Features)
        """
        out = torch.cat([block(x) for block in self.blocks], dim=-1)
        return out

    def fit(self, x: torch.Tensor) -> None:
        """Obtain parameter valies from the first available example.

        Accepts either a single example or a batch as an input.

        Args:
            x: Tensor of shape (Channels * Timeobs) or
                Tensor of shape (Batch * Channels * Timeobs)
        """
        for block in self.blocks:
            block.fit(x)

    @property
    def is_fitted(self) -> bool:
        """Indicates if module biases were fitted to data."""
        return self.blocks[0].is_fitted

    @property
    def feature_names(self) -> list:
        """(pattern, dilation, channels, threshold) tuples to identify features."""
        assert self.is_fitted, "module needs to be fitted for thresholds to be named"
        feature_names = []
        for block in self.blocks:
            feature_names += block.feature_names
        return feature_names

    @property
    def in_channels(self) -> int:
        """Number of channels in each timeseries."""
        return self.blocks[0].in_channels

    @property
    def combination_order(self) -> int:
        """The maximum number of channels to be combined."""
        return self.blocks[0].mix.order

    @property
    def device(self) -> torch.device:
        """The device the module is loaded on."""
        return next(self.parameters()).device