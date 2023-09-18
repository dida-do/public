# X-ROCKET code repository

To use the X-rocket encoder for timeseries embeddings install the dependencies in `requirements.txt` and import as follows:
```python
from xrocket.encoder import XRocket
```

Then initialize the encoder with the desired hyperparameters:
```python
XRocket(
    in_channels: int,
    max_kernel_span: int,
    combination_order: int = 1,
    combination_method: str = "additive",
    feature_cap: int = 10_000,
    kernel_length: int = 9,
    max_dilations: int = 32,
)
```

The following hyperparameters can be chosen:
- in_channels: The number of channels in the data.
- max_kernel_span: Maximum length to be considered for patter search,
    usually set to the number of time-observations in a typical timeseries.
- combination_order: The maximum number of channels to be interacted, default=1.
- combination_method: Keyword for the channel mixing method, default='additive'.
- feature_cap: Maximum number of embedding values to be considered, default=10,000.
- kernel_length: The length of the 1D convolutional kernels, default=9.
- max_dilations: The maximum number of distinct dilation values, default=32.

If the encoder thresholds are not explicitly fit to a data example before encoding, the first example will automatically define the thresholds.