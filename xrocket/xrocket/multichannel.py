import torch
from torch import nn

class ChannelMix(nn.Module):
    """Channel mixing layer for ROCKET transformation of timeseries.

    This layer serves to select and interact the input channels to create uni-
    and multivariate feature sequences.
    I.e., a forward pass transforms a tensor of shape
    (Batch * Kernels * Channels * Timeobs) into a tensor of shape
    (Batch * Kernels * Combinations * Timeobs).

    This implementation is based on the descriptions in:
    Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
    "Minirocket: A very fast (almost) deterministic transform for time series classification."
    Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

    In contrast to the paper, all channel combinations will be considered, not only
    a randomly selected subset as in the original authors' implementation.

    Attributes:
        in_channels: The number of channels in the data.
        in_kernels: The number of distinct kernels passed to the module.
        order: The maximum number of channels to be interacted.
        method: The channel mixing method, either 'additive' or 'multiplicative'.
        combinations: List of channel combinations being interacted.
        weight: List of parameters for each combination.
        num_combinations: The number of channel combinations considered in the module.
    """

    def __init__(
        self,
        in_channels: int,
        in_kernels: int,
        order: int = 1,
        method: str = "additive",
    ) -> None:
        """Set up attributes including combinations and combination weights for the layer.

        Args:
            in_channels: The number of channels going into the module.
            in_kernels: The number of kernel outputs going into the module.
            order: The maximum number of channels to be interacted.
            method: Keyword to indicate the channel mixing method, default='additive'.
        """
        super().__init__()
        self.in_channels = in_channels
        self.order = order
        self.method = method
        self._initialize_weights(in_kernels=in_kernels)

    def _initialize_weights(self, in_kernels: int) -> None:
        """Set up the channel combinations as weights.

        The created weight tensor will have dimensions
        (Kernels * Combinations * Channels).

        Args:
            in_kernels: The number of kernel outputs going into the module.
        """
        weight = torch.Tensor([])
        for order in range(self.order):
            combinations = torch.combinations(torch.arange(self.in_channels), order + 1)
            channel_map = torch.zeros(len(combinations), self.in_channels).scatter(
                1, combinations, 1
            )
            weight = torch.cat([weight, channel_map])
        weight = weight.unsqueeze(0).repeat(in_kernels, 1, 1)
        self.weight = nn.Parameter(data=weight, requires_grad=False)

    @property
    def num_combinations(self) -> int:
        """The number of channel combinations to be considered."""
        return self.weight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass to calculate comnbination-wise activations.

        All channel combinations up to the initialized order will be included
        (This deviates from the original authors who use only a random selection).
        Channel activations can be interacted additively or multiplicatively to
        also capture negative correlations (The original authors suggest an
        additive implementation, which is the default here and runs faster).

        Args:
            x: Tensor of shape (Batch * Kernels * Channels * Timeobs)

        Returns:
            out: Tensor of shape (Batch * Kernels * Combinations * Timeobs)
        """
        if self.method == "additive":
            x = self.weight.matmul(x)
        elif self.method == "multiplicative":
            x = x[:, :, None, :, :].mul(self.weight[None, :, :, :, None])
            x = x.where(x != 0, torch.ones(1, device=x.device)).prod(dim=-2)
        else:
            raise ValueError(f"method '{self.method}' not available")

        return x

    @property
    def combinations(self) -> list:
        """A list of the channel weightings considered."""
        return self.weight.reshape(-1, self.in_channels).tolist()
