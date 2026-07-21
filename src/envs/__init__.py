"""Numpy 2.0 removed several deprecated type aliases (`np.float_`, `np.int_`,
etc). gymnasium 0.29.1's classic_control envs (Acrobot, MountainCar) still
reference `np.float_` internally, which crashes on `numpy>=2.0`. Restore the
alias here, once, before any CARL/gym env is constructed - this is a
well-known, minimal numpy2-migration shim, not a bug in our own code.
"""
import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64
