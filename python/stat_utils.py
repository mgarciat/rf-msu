"""
Information-theoretic utilities for integer-encoded categorical data.

This module provides functions for computing entropy,
conditional entropy, mutual information, symmetrical
uncertainty, and contingency tables.

Functions accept NumPy arrays or compatible sequences. All entropy and
information values are measured in bits. Codes are expected to be nonnegative
integers; inputs are generally not validated by this module.

Authors
-------
- Miguel Garcia Torres <mgarciat@upo.es>

Affiliation
-----------
Data Science & Big Data Lab
Universidad Pablo de Olavide, Spain

License
-------
See the LICENSE file in the project repository.

"""

import numpy as np
from typing import Sequence, Optional

def log2(x):
    """Return the base-2 logarithm elementwise.

    Parameters
    ----------
    x : array_like
        Positive input values.

    Returns
    -------
    scalar or numpy.ndarray
        Base-2 logarithms with the same broadcasting behavior as ``numpy.log2``.

    Notes
    -----
    ``x <= 0`` produces the usual NumPy warnings and nonfinite results.
    """
    return np.log2(x)

def xlogx(X: np.ndarray) -> np.ndarray:
    """Compute ``x * ln(x)`` elementwise, defining the result as zero for nonpositive inputs.

    Parameters
    ----------
    X : array_like
        Numeric scalar or array, conventionally containing nonnegative counts.

    Returns
    -------
    numpy.ndarray or numpy scalar
        Values of ``X * ln(X)`` for positive entries, zero otherwise.

    Notes
    -----
    Uses natural logarithms. Negative inputs are treated as zero rather than rejected.
    """
    a = np.asarray(X, dtype=float)
    # Faster than clipping for the tiny histograms used heavily by tree split
    # scoring. It keeps the same 0 * log(0) = 0 convention.
    out = np.zeros_like(a, dtype=float)
    mask = a > 0.0
    out[mask] = a[mask] * np.log(a[mask])
    return out

def contingency_table_nvalues(X: Sequence[int],
                              nxvalues: int,
                              Y: Sequence[int],
                              nyvalues: int) -> np.ndarray:
    """Count paired discrete observations in a fixed-size contingency table.

    Parameters
    ----------
    X : sequence of int
        Row-category codes in the range ``[0, nxvalues - 1]``.
    nxvalues : int
        Number of allowed row categories.
    Y : sequence of int
        Column-category codes in the range ``[0, nyvalues - 1]``.
    nyvalues : int
        Number of allowed column categories.

    Returns
    -------
    numpy.ndarray
        Integer count matrix with shape ``(nxvalues, nyvalues)``.

    Notes
    -----
    Both inputs must have the same length and contain nonnegative integer codes within the specified ranges. These assumptions are not explicitly validated.
    """
    dtype = np.int64

    xa = np.asarray(X).ravel()
    ya = np.asarray(Y).ravel()

    # ravel indices into single dimension and count
    idx = xa * np.int64(nyvalues) + ya
    counts = np.bincount(idx, minlength=nxvalues * nyvalues)

    # reshape back to 2D (rows = x values, cols = y values)
    table = counts.reshape((nxvalues, nyvalues)).astype(dtype, copy=False)
    return table

def contingency_table(X: Sequence[int],
                      Y: Sequence[int]) -> np.ndarray:
    """Build a contingency table while inferring category counts from the data.

    Parameters
    ----------
    X : sequence of int
        Nonnegative integer row-category codes.
    Y : sequence of int
        Nonnegative integer column-category codes of equal length.

    Returns
    -------
    numpy.ndarray
        Count matrix with inferred numbers of rows and columns.

    Notes
    -----
    Category counts are inferred as ``max(code) + 1`` using ``numpy.bincount``. Missing intermediate category codes produce empty rows/columns.
    """
    xvalues = np.bincount(X).shape[0]
    yvalues = np.bincount(Y).shape[0]

    return contingency_table_nvalues(X, xvalues, Y, yvalues)



def entropy_fromhistogram(hist: Sequence[float], *, dtype: Optional[np.dtype] = None) -> float:
    """Compute Shannon entropy in bits from unnormalized category counts.

    Parameters
    ----------
    hist : sequence of float
        Histogram containing nonnegative frequencies or weights.
    dtype : numpy.dtype, optional
        Internal numeric dtype; defaults to ``numpy.float64``.

    Returns
    -------
    float
        Shannon entropy in bits; zero when the histogram sum is zero.

    Notes
    -----
    Count values are expected to be nonnegative. No explicit validation is performed.
    """
    if dtype is None:
        dtype = np.float64

    h = np.asarray(hist, dtype=dtype).ravel()
    total = float(h.sum())
    if total == 0.0:
        return 0.0

    # numerator = - sum_i x log x + (total log total)
    numerator = -float(xlogx(h).sum()) + float(xlogx(total))
    # divide by (total * ln(2)) to convert to bits and to normalize
    return float(numerator / (total * np.log(2.0)))

def entropy_nvalues(X: Sequence[int], nxvalues: int) -> float:
    """Compute Shannon entropy in bits for integer category codes.

    Parameters
    ----------
    X : sequence of int
        Nonnegative integer category codes.
    nxvalues : int
        Number of categories (codes should lie below this value).

    Returns
    -------
    float
        Entropy of ``X`` in bits, or zero for an empty array.

    Notes
    -----
    ``nxvalues`` controls the minimum histogram length, not a validated upper bound.
    """
    xa = np.asarray(X, dtype=np.int64).ravel()
    if xa.size == 0:
        return 0.0
    # bincount handles empty bins automatically with minlength
    hist = np.bincount(xa, minlength=nxvalues).astype(np.float64, copy=False)
    return entropy_fromhistogram(hist)

def entropy(X: Sequence[int]) -> float:
    """Compute Shannon entropy in bits with automatically inferred categories.

    Parameters
    ----------
    X : sequence of int
        Nonnegative integer category codes.

    Returns
    -------
    float
        Entropy of ``X`` in bits.

    Notes
    -----
    An empty ``X`` produces zero; negative codes are not supported.
    """
    xvalues = np.bincount(X).shape[0]

    return entropy_nvalues(X, xvalues)

def conditional_entropy_onrows(ctable: Sequence[Sequence[float]]) -> float:
    """Compute ``H(Y | X)`` in bits from a contingency table.

    Parameters
    ----------
    ctable : array_like of shape (n_x, n_y)
        Nonnegative joint-frequency matrix; rows represent ``X`` and columns ``Y``.

    Returns
    -------
    float
        Conditional Shannon entropy ``H(Y | X)`` in bits; zero for a zero-total table.

    Notes
    -----
    Input frequencies are assumed to be nonnegative.
    """
    M = np.asarray(ctable, dtype=float)
    if M.size == 0:
        return 0.0
    row_sums = M.sum(axis=1)
    total = float(row_sums.sum())
    if total == 0.0:
        return 0.0
    numerator = -float(xlogx(M).sum()) + float(xlogx(row_sums).sum())
    return float(numerator / (total * np.log(2.0)))



def conditional_entropy_nvalues(X: Sequence[int], nxvalues: int, Y: Sequence[int], nyvalues: int) -> float:
    """Compute ``H(X | Y)`` in bits from paired integer-coded samples.

    Parameters
    ----------
    X : sequence of int
        Target category codes.
    nxvalues : int
        Number of possible categories in ``X``.
    Y : sequence of int
        Conditioning category codes.
    nyvalues : int
        Number of possible categories in ``Y``.

    Returns
    -------
    float
        Conditional entropy ``H(X | Y)`` in bits.

    Notes
    -----
    The contingency table is formed with rows representing ``Y`` and columns ``X``.
    """
    ctable = contingency_table_nvalues(Y, nyvalues, X, nxvalues).astype(float, copy=False)
    return conditional_entropy_onrows(ctable)

def conditional_entropy(X: Sequence[int], Y: Sequence[int]) -> float:
    """Compute ``H(X | Y)`` in bits with inferred category counts.

    Parameters
    ----------
    X : sequence of int
        Target categorical variable, encoded as nonnegative integers.
    Y : sequence of int
        Conditioning variable, encoded as nonnegative integers.

    Returns
    -------
    float
        Conditional entropy ``H(X | Y)`` in bits.

    Notes
    -----
    ``X`` and ``Y`` must have matching lengths.
    """
    xvalues = np.bincount(X).shape[0]
    yvalues = np.bincount(Y).shape[0]

    return conditional_entropy_nvalues(X, xvalues, Y, yvalues)

def joint_entropy_nvalues(X: Sequence[int], nxvalues: int,
                          Y: Sequence[int], nyvalues: int) -> float:
    """Compute joint Shannon entropy ``H(X, Y)`` in bits.

    Parameters
    ----------
    X : sequence of int
        First categorical variable.
    nxvalues : int
        Number of possible ``X`` categories.
    Y : sequence of int
        Second categorical variable.
    nyvalues : int
        Number of possible ``Y`` categories.

    Returns
    -------
    float
        Joint entropy in bits, calculated as ``H(X | Y) + H(Y)``.

    Notes
    -----
    The input arrays must contain nonnegative integer codes and be equally long.
    """
    histogram = np.bincount(Y, minlength=nyvalues)
    return conditional_entropy_nvalues(X, nxvalues, Y, nyvalues) + entropy_fromhistogram(histogram)    



def mutual_information_nvalues(X: Sequence[int], nxvalues: int, Y: Sequence[int], nyvalues: int) -> float:
    """Compute mutual information ``I(X; Y)`` in bits.

    Parameters
    ----------
    X : sequence of int
        First categorical variable.
    nxvalues : int
        Number of possible ``X`` categories.
    Y : sequence of int
        Second categorical variable.
    nyvalues : int
        Number of possible ``Y`` categories.

    Returns
    -------
    float
        Mutual information ``H(X) - H(X | Y)`` in bits.

    Notes
    -----
    Finite-precision arithmetic can yield small negative values near theoretical zero.
    """
    Hx = entropy_nvalues(X, nxvalues)
    Hxy = conditional_entropy_nvalues(X, nxvalues, Y, nyvalues)

    return Hx - Hxy

def mutual_information(X: Sequence[int], Y: Sequence[int]) -> float:
    """Compute mutual information in bits with inferred category counts.

    Parameters
    ----------
    X : sequence of int
        First nonnegative integer-coded variable.
    Y : sequence of int
        Second nonnegative integer-coded variable.

    Returns
    -------
    float
        Mutual information ``I(X; Y)`` in bits.

    Notes
    -----
    Both input sequences must have equal lengths.
    """
    xvalues = np.bincount(X).shape[0]
    yvalues = np.bincount(Y).shape[0]
    
    return mutual_information_nvalues(X, xvalues, Y, yvalues)

def symmetrical_uncertainty_nvalues(X: Sequence[int], nxvalues: int, Y: Sequence[int], nyvalues: int) -> float:
    """Compute normalized symmetrical uncertainty between two categorical variables.

    Parameters
    ----------
    X : sequence of int
        First categorical variable.
    nxvalues : int
        Number of ``X`` categories.
    Y : sequence of int
        Second categorical variable.
    nyvalues : int
        Number of ``Y`` categories.

    Returns
    -------
    float
        ``2 * I(X; Y) / (H(X) + H(Y))``.
    """

    I = mutual_information_nvalues(X, nxvalues, Y, nyvalues)
    Hx = entropy_nvalues(X, nxvalues)
    Hy = entropy_nvalues(Y, nyvalues)

    denominator = Hx + Hy

    if denominator == 0:
        return 0.0
    
    return 2 * I / denominator

def symmetrical_uncertainty(X: Sequence[int], Y: Sequence[int]) -> float:
    """Compute symmetrical uncertainty with automatically inferred categories.

    Parameters
    ----------
    X : sequence of int
        First nonnegative integer-coded variable.
    Y : sequence of int
        Second nonnegative integer-coded variable.

    Returns
    -------
    float
        Normalized symmetrical uncertainty between ``X`` and ``Y``.

    Notes
    -----
    The implementation yields ``nan`` when both variables are constant.
    """
    xvalues = np.bincount(X).shape[0]
    yvalues = np.bincount(Y).shape[0]
    
    return symmetrical_uncertainty_nvalues(X, xvalues, Y, yvalues)


