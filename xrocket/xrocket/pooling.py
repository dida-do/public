import math
import warnings
import torch
from torch import nn

class PPVThresholds(nn.Module):
    """Threshold layer for ROCKET transformation of timeseries.

    This layer serves to apply proportion of positive values pooling based on a
    set of threshold values from the convolutional outputs.
    I.e., a forward pass transforms a tensor of shape
    (Batch * Kernels * Combinations * Timeobs) into a tensor of shape
    (Batch * Kernels * Combinations * Thresholds).
    LAyer needs to be fitted to define the values of the thresholds.

    This implementation conforms to the descriptions in:
    Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
    "Minirocket: A very fast (almost) deterministic transform for time series classification."
    Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

    Attributes:
        num_thresholds: The number of thresholds per channel.
        is_fitted: Indicates that thresholds are fitted to a data example.
        bias: Tensor of shape (Kernels * Dilations * Combinations * Quantiles)
    """

    def __init__(
        self,
        num_thresholds: int,
    ) -> None:
        """Set up attributes including quantile values for the layer.

        Args:
            num_thresholds: The number of thresholds to be considered per channel.
        """
        super().__init__()
        self.num_thresholds = num_thresholds
        self.is_fitted: bool = False

    def _select_quantiles(
        self,
        num_thresholds: int,
        uniform: bool = False,
    ) -> torch.Tensor:
        """Automatically selects the quantile values to initialize threshold values.

        Following the original authors' code, the module uses a "low-discrepancy
        sequence to assign quantiles to kernel/dilation combinations", source:
        https://github.com/angus924/minirocket/blob/main/code/minirocket_multivariate.py

        Alternatively, quantiles can be choosen to be uniformly spaced in [0, 1].

        Args:
            num_thresholds: The number of thresholds to be considered per channel.
            uniform: Indicates if quantiles should be uniformly spaced, default=False.
        """
        if uniform:
            # uniformly spaced quantiles
            quantiles = torch.linspace(0, 1, num_thresholds + 2)[1:-1]
        else:
            # low-discrepancy sequence to assign quantiles 
            phi = (math.sqrt(5) + 1) / 2
            quantiles = torch.Tensor(
                [((i + 1) * phi) % 1 for i in range(num_thresholds)]
            )

        return quantiles

    def fit(self, x: torch.Tensor, quantiles: list = None) -> None:
        """Obtain quantile values from the first available example to use as thresholds.

        Accepts either a single example or a batch as an input.

        Args:
            x: Tensor of shape (Batch * Kernels * Combinations * Timeobs)
            quantiles (optional): A list of values between 0 and 1 to indicate the quantiles
                at which to set the thresholds.
        """
        # get quantiles
        if quantiles is None:
            quantiles = self._select_quantiles(
                num_thresholds=self.num_thresholds, uniform=False
            )
        else:
            if type(quantiles) != list or not all(0 <= q <= 1 for q in quantiles):
                raise ValueError(
                    "quantiles needs to be a list of values between 0 and 1"
                )

        # flatten if input is a batch
        if len(x.shape) == 4:
            x = x.movedim(0, -1).flatten(start_dim=-2)

        # extract threshold values from activation quantiles
        thresholds = x.quantile(q=quantiles.to(x.device), dim=-1).movedim(
            source=0, destination=-1
        )

        # set attributes
        self.bias = nn.Parameter(data=thresholds, requires_grad=False)
        self.is_fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass to calculate pooled features.

        Module weights will be fit to the first example if not yet fitted.

        Args:
            x: Tensor of shape (Batch * Kernels * Combinations * Timeobs)

        Returns:
            out: Tensor of shape (Batch * Kernels * Combinations * Thresholds)
        """
        # fit if module is not yet fitted
        if not self.is_fitted:
            self.fit(x)
            warnings.warn("automatically fit biases to first training example")

        # add missing dims
        thresholds = self.bias.unsqueeze(-1)
        x = x.unsqueeze(-2)

        # apply percentage of values pooling with thresholds
        x = x.gt(thresholds).sum(dim=-1).div(x.size(-1))
        return x

    @property
    def thresholds(self) -> list:
        """A list of the threshold values considered."""
        return self.bias.flatten().tolist()
