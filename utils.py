# Copyright 2026 triloy8
#
# SPDX-License-Identifier: Apache-2.0

import torch

# Taken from https://github.com/pytorch/pytorch/issues/50334#issuecomment-2304751532
def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int=-1, extrapolate: str='constant') -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)
    
    m = torch.diff(fp) / torch.diff(xp) # slope
    b = fp[..., :-1] - m * xp[..., :-1] # offset
    indices = torch.searchsorted(xp, x, right=False)
    
    if extrapolate == 'constant':
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else: # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)
    
    return values.movedim(-1, dim)