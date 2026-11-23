"""Reuse a pretrained planner projection for prefix-pruned BEV features.

The frozen compressor retains prefixes independently in each deblock, then
concatenates those deblocks. It does not retain one prefix of the concatenated
feature. Selecting the same input columns in the first pretrained convolution
is equivalent to zero-expanding the removed channels before that projection.
"""

import copy


FIRST_PROJECTION_WEIGHT = "layers.0.double_conv.0.weight"


def retained_projection_indices(source_widths, retained_widths):
    """Return retained source-channel indices in deblock concatenation order."""
    if (not isinstance(source_widths, (list, tuple))
            or not isinstance(retained_widths, (list, tuple))
            or not source_widths or len(source_widths) != len(retained_widths)):
        raise ValueError("source and retained deblock widths must have matching nonempty lengths")
    indices = []
    offset = 0
    for source, retained in zip(source_widths, retained_widths):
        if (type(source) is not int or type(retained) is not int
                or source <= 0 or retained <= 0 or retained > source):
            raise ValueError("retained deblock widths must be positive integers not exceeding source")
        indices.extend(range(offset, offset + retained))
        offset += source
    return indices


def slice_pretrained_projection(config, weights, source_widths, retained_widths):
    """Copy the trained projection, slicing only its first input-channel axis."""
    indices = retained_projection_indices(source_widths, retained_widths)
    source_channels = sum(source_widths)
    if config.get("input_dim") != source_channels:
        raise ValueError("planner projection input_dim does not match source deblock widths")
    kernel = weights.get(FIRST_PROJECTION_WEIGHT)
    if kernel is None or kernel.ndim != 4 or kernel.shape[1] != source_channels:
        raise ValueError("pretrained first projection kernel does not match source channel contract")
    adapted_config = copy.deepcopy(config)
    adapted_config["input_dim"] = sum(retained_widths)
    adapted_weights = {key: tensor.clone() for key, tensor in weights.items()}
    adapted_weights[FIRST_PROJECTION_WEIGHT] = kernel[:, indices, :, :].clone()
    return adapted_config, adapted_weights
