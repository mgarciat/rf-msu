"""Multivariate Shannon entropy and symmetrical uncertainty utilities.

Functions operate on integer-coded discrete features; joint entropy and
MSU use logarithms in base two. Original computational logic is preserved.

Notes
-----
This module imports ``stat_util``, while the companion module in this project
is named ``stat_utils``. Keep an alias/module with that name or adjust the
import in your packaging configuration.

Authors
-------
- Miguel Garcia Torres <mgarciat@upo.es>
- Thomas Vanhaeren

License
-------
See the project LICENSE file.
"""

import numpy as np
import stat_utils as ul
from typing import Sequence


def joint_entropy_nvalues(X: Sequence[int], nxvalues: int,
                          Y: Sequence[int], nyvalues: int) -> float:
    """Compute joint Shannon entropy for two discrete variables.
    
    Parameters
    ----------
    X, Y : sequence of int
        Paired, nonnegative integer-encoded observations.
    nxvalues, nyvalues : int
        Number of possible categories in each variable.
    
    Returns
    -------
    float
        Joint entropy H(X, Y), measured in bits.
    
    Notes
    -----
    Builds a contingency table via ``stat_util`` and computes its entropy."""
    ctable = ul.contingency_table_nvalues(X, nxvalues, Y, nyvalues)
    return ul.entropy_fromhistogram(ctable.ravel())


def concatenate_features(X: Sequence[int], Y: Sequence[int], A: np.ndarray | None = None) -> int:
    """Encode observed pairs of categorical values with consecutive integers.
    
    Parameters
    ----------
    X, Y : sequence of int
        Equally long vectors of discrete category codes.
    A : numpy.ndarray or None, optional
        Output vector written in place; if omitted, the internal vector is not returned.
    
    Returns
    -------
    int
        Number of unique observed pairs.
    
    Raises
    ------
    ValueError
        If input lengths or output vector length differ.
    
    Notes
    -----
    Codes are assigned in order of first occurrence starting at zero.
    Use a preallocated ``A`` to retrieve the resulting codes."""
    X = np.asarray(X, dtype=np.int64).ravel()
    Y = np.asarray(Y, dtype=np.int64).ravel()

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X e Y deben tener la misma longitud")

    if A is None:
        A = np.empty(X.shape[0], dtype=np.int64)
    else:
        if len(A) != len(X):
            raise ValueError("A debe tener la misma longitud que X e Y")

    values = {}
    index = 0

    for i in range(X.shape[0]):
        key = (int(X[i]), int(Y[i]))
        if key not in values:
            values[key] = index
            A[i] = index
            index += 1
        else:
            A[i] = values[key]

    return len(values)


def joint_entropy_for_row_features_X(tdataset: np.ndarray,
                                   nvalues: Sequence[int],
                                   features: Sequence[int]) -> float:
    """Compute joint entropy of selected rows of a discrete data matrix.
    
    Parameters
    ----------
    tdataset : numpy.ndarray of shape (n_features, n_samples)
        Integer-encoded categorical observations with features as rows.
    nvalues : sequence of int
        Number of possible categories for each row.
    features : sequence of int
        Indices of selected feature rows; must not be empty.
    
    Returns
    -------
    float
        Joint Shannon entropy of selected variables in bits.
    
    Raises
    ------
    ValueError
        If no features are selected.
    
    Notes
    -----
    Concatenates observed category tuples when at least three features are
    selected. Certain all-constant paths use a special case; compare with
    ``_joint_entropy_columns`` for independent mathematical validation."""
    tdataset = np.asarray(tdataset)
    nvalues = np.asarray(nvalues, dtype=np.int64)
    features = np.asarray(features, dtype=np.int64)

    if features.size == 0:
        raise ValueError("features no puede estar vacío")

    if features.size == 1:
        feature = features[0]
        return ul.entropy_nvalues(tdataset[feature], int(nvalues[feature]))

    elif features.size == 2:
        feature1 = features[0]
        feature2 = features[1]
        return ul.joint_entropy_nvalues(
            tdataset[feature1], int(nvalues[feature1]),
            tdataset[feature2], int(nvalues[feature2])
        )

    feature0 = features[0]
    feature1 = features[1]

    sumMJE = 0.0
    XY = np.empty(tdataset.shape[1], dtype=np.int64)

    if nvalues[feature0] == 1 and nvalues[feature1] == 1:
        nXY = 1
        XY = tdataset[feature0].copy()
    else:
        nXY = concatenate_features(tdataset[feature0], tdataset[feature1], XY)

    for j in range(2, features.size):
        featurej = features[j]

        if j == features.size - 1:
            #if nXY == 1:
            #    sumMJE = 0.0
            #else:
            sumMJE = joint_entropy_nvalues(
                tdataset[featurej], int(nvalues[featurej]),
                XY, int(nXY)
            )

        A = np.empty(XY.shape[0], dtype=np.int64)

        if nXY == 1 and nvalues[featurej] == 1:
            nXY = 1
        else:
            nXY = concatenate_features(tdataset[featurej], XY, A)
            XY = A.copy()

    return sumMJE

def joint_entropy_for_row_features_XY(tdataset: np.ndarray,
                                      nvalues: Sequence[int],
                                      features: Sequence[int],
                                      Y: Sequence[int],
                                      nvaluesY: int) -> float:
    """Compute joint entropy of selected feature rows and a target variable.
    
    Parameters
    ----------
    tdataset : numpy.ndarray of shape (n_features, n_samples)
        Integer-encoded categorical feature rows.
    nvalues : sequence of int
        Cardinalities of feature rows.
    features : sequence of int
        Selected row indices; must not be empty.
    Y : sequence of int
        Extra categorical target variable.
    nvaluesY : int
        Number of allowed categories in ``Y``.
    
    Returns
    -------
    float
        Joint entropy H(X_selected, Y) in bits.
    
    Raises
    ------
    ValueError
        If ``features`` is empty."""
    tdataset = np.asarray(tdataset)
    nvalues = np.asarray(nvalues, dtype=np.int64)
    features = np.asarray(features, dtype=np.int64)
    Y = np.asarray(Y, dtype=np.int64).ravel()

    if features.size == 0:
        raise ValueError("features no puede estar vacío")

    if features.size == 1:
        feature = features[0]
        return ul.joint_entropy_nvalues(tdataset[feature], int(nvalues[feature]), Y, nvaluesY)

    elif features.size == 2:
        feature0 = features[0]
        feature1 = features[1]

        XY = np.empty(Y.shape[0], dtype=np.int64)
        nXY = concatenate_features(tdataset[feature0], tdataset[feature1], XY)

        return ul.joint_entropy_nvalues(Y, nvaluesY, XY, nXY)

    # Caso general: concatenar sucesivamente todas las features indicadas
    feature0 = features[0]
    feature1 = features[1]

    XY = np.empty(Y.shape[0], dtype=np.int64)
    nXY = concatenate_features(tdataset[feature0], tdataset[feature1], XY)

    for j in range(2, features.size):
        featurej = features[j]
        A = np.empty(Y.shape[0], dtype=np.int64)
        nXY = concatenate_features(tdataset[featurej], XY, A)
        XY = A.copy()

    return ul.joint_entropy_nvalues(Y, nvaluesY, XY, nXY)


def multivariate_symmetrical_uncertainty_Xy(tdata: np.ndarray,
                                            nvalues: Sequence[int],
                                            features: Sequence[int],
                                            YY: Sequence[int],
                                            nvaluesY: int) -> float:
    """Compute multivariate symmetrical uncertainty of features plus target.
    
    Parameters
    ----------
    tdata : numpy.ndarray of shape (n_features, n_samples)
        Discretized feature rows.
    nvalues : sequence of int
        Cardinalities of each feature.
    features : sequence of int
        Indices of selected features.
    YY : sequence of int
        Discrete target observations.
    nvaluesY : int
        Number of target categories.
    
    Returns
    -------
    float
        Normalized total correlation over selected features and target.
    
    Notes
    -----
    For n = len(features)+1, returns n/(n-1) times the ratio of total
    correlation to the sum of marginal entropies. Returns 0 when that sum
    is zero. Requires at least one selected feature to avoid n=1 division."""
    tdata = np.asarray(tdata)
    nvalues = np.asarray(nvalues, dtype=np.int64)
    features = np.asarray(features, dtype=np.int64)
    YY = np.asarray(YY, dtype=np.int64).ravel()

    HY = ul.entropy_nvalues(YY, nvaluesY)
    HX1nY = joint_entropy_for_row_features_XY(tdata, nvalues, features, YY, nvaluesY)

    sumHX1nY = HY
    for f in features:
        Hf = ul.entropy_nvalues(tdata[f], int(nvalues[f]))
        sumHX1nY += Hf

    n = features.size + 1
    num = sumHX1nY - HX1nY
    den = sumHX1nY

    if den == 0:
        return 0.0
    else:
        return n / (n - 1) * (num / den)


def multivariate_symmetrical_uncertainty_X(tdata: np.ndarray,
                                           nvalues: Sequence[int],
                                           features: Sequence[int]) -> float:
    """Compute multivariate symmetrical uncertainty for selected features.
    
    Parameters
    ----------
    tdata : numpy.ndarray of shape (n_features, n_samples)
        Discretized feature rows.
    nvalues : sequence of int
        Category counts for each feature.
    features : sequence of int
        Selected feature indices.
    
    Returns
    -------
    float
        Normalized multivariate symmetrical uncertainty in bits ratio form.
    
    Notes
    -----
    Returns 0 for exactly one feature or zero total marginal entropy.
    Empty feature collections are not explicitly validated."""
    features = np.asarray(features, dtype=np.int64)
    nvalues = np.asarray(nvalues, dtype=np.int64)
    tdata = np.asarray(tdata)

    if features.size == 1:
        return 0.0

    HX1nY = joint_entropy_for_row_features_X(tdata, nvalues, features)
    sumHX1nY = 0.0

    for f in features:
        Hf = ul.entropy_nvalues(tdata[f], int(nvalues[f]))
        sumHX1nY += Hf

    n = features.size
    num = sumHX1nY - HX1nY
    den = sumHX1nY

    if den == 0:
        return 0.0
    else:
        return n / (n - 1) * (num / den)


def _nvalues_per_feature(tdata: np.ndarray) -> np.ndarray:
    """Infer categorical cardinalities for each row of a transposed data matrix.
    
    Parameters
    ----------
    tdata : numpy.ndarray of shape (n_features, n_samples)
        Nonnegative integer-encoded feature values.
    
    Returns
    -------
    numpy.ndarray of int
        ``max(code) + 1`` for each feature; unobserved categories still count."""
    return np.array([np.bincount(tdata[i]).shape[0] for i in range(tdata.shape[0])], dtype=np.int64)


def msuXy_legacy(Xd: np.ndarray, y: np.ndarray, features: Sequence[int]) -> float:
    """Compute feature-target MSU using the original row-based implementation.
    
    Parameters
    ----------
    Xd : numpy.ndarray of shape (n_samples, n_features)
        Discrete feature matrix with nonnegative integer codes.
    y : numpy.ndarray of shape (n_samples,)
        Discrete target labels.
    features : sequence of int
        Feature indices.
    
    Returns
    -------
    float
        Multivariate symmetrical uncertainty for selected features and target;
        zero when no features are selected.
    
    Notes
    -----
    Computes cardinalities for every feature, even unselected ones."""
    if len(features) == 0:
        return 0.0

    Xdt = Xd.T  # (n_features, n_samples)
    nvalues = _nvalues_per_feature(Xdt)
    nvaluesY = np.bincount(y).shape[0]

    return multivariate_symmetrical_uncertainty_Xy(tdata=Xdt, nvalues=nvalues, features=np.asarray(features, dtype=np.int64), YY=y, nvaluesY=nvaluesY)


# OLD PRODUCTION CODE, kept above as msuXy_legacy for notebook benchmarks:
# def msuXy(Xd: np.ndarray, y: np.ndarray, features: Sequence[int]) -> float:
#     if len(features) == 0:
#         return 0.0
#
#     Xdt = Xd.T
#     nvalues = _nvalues_per_feature(Xdt)
#     nvaluesY = np.bincount(y).shape[0]
#
#     return multivariate_symmetrical_uncertainty_Xy(
#         tdata=Xdt,
#         nvalues=nvalues,
#         features=np.asarray(features, dtype=np.int64),
#         YY=y,
#         nvaluesY=nvaluesY,
#     )
#
# NEW CODE:
# msuXy below computes the same score but only touches the requested trial
# subset instead of recounting all columns in the full matrix.
#
# NUMBA/JOBLIB NOTE:
# This function deliberately stays NumPy-based. Its hot operation is
# np.unique(..., axis=0), which Numba does not currently accelerate cleanly for
# this 2D use case, and each MSU candidate is too small for joblib scheduling to
# help. Numba is used in upo_disc.py for transform(X), and joblib is used there
# for optional per-column discretization.
def _joint_entropy_columns(cols: np.ndarray) -> float:
    """Compute Shannon entropy of a vector or a matrix of joint observations.
    
    Parameters
    ----------
    cols : numpy.ndarray
        One-dimensional categorical codes, or matrix of shape
        ``(n_samples, n_variables)`` with integer-coded columns.
    
    Returns
    -------
    float
        Empirical Shannon entropy of observed joint outcomes in bits.
    
    Notes
    -----
    For matrices, uses ``numpy.unique(..., axis=0)``; empty rows have zero
    entropy."""
    cols = np.asarray(cols, dtype=np.int64)
    if cols.ndim == 1:
        nvalues = int(cols.max()) + 1 if cols.size else 0
        return ul.entropy_nvalues(cols, nvalues)

    if cols.shape[0] == 0:
        return 0.0

    _, counts = np.unique(cols, axis=0, return_counts=True)
    return ul.entropy_fromhistogram(counts)


def msuXy(Xd: np.ndarray, y: np.ndarray, features: Sequence[int]) -> float:
    """Compute feature-target multivariate symmetrical uncertainty efficiently.
    
    Parameters
    ----------
    Xd : numpy.ndarray of shape (n_samples, n_features)
        Matrix of discrete, nonnegative integer-encoded attributes.
    y : numpy.ndarray of shape (n_samples,)
        Integer-encoded target variable.
    features : sequence of int
        Indices of attributes to include in the score.
    
    Returns
    -------
    float
        Normalized total correlation of selected features and target; returns
        zero if no attributes are selected or marginal entropy sum is zero.
    
    Raises
    ------
    ValueError
        If ``Xd`` and ``y`` contain different numbers of instances.
    
    Notes
    -----
    Processes only selected columns using ``numpy.unique`` to estimate joint
    entropy and avoids recomputing per-feature cardinalities for all columns."""
    features = np.asarray(features, dtype=np.int64)
    if features.size == 0:
        return 0.0

    Xd = np.asarray(Xd, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64).ravel()

    if Xd.shape[0] != y.shape[0]:
        raise ValueError("Xd e y deben tener el mismo numero de instancias.")

    subset = Xd[:, features]
    y_col = y.reshape(-1, 1)

    hy = _joint_entropy_columns(y)
    h_joint = _joint_entropy_columns(np.hstack([subset, y_col]))

    sum_h = hy
    for f in features:
        sum_h += _joint_entropy_columns(Xd[:, int(f)])

    if sum_h == 0.0:
        return 0.0

    n_variables = features.size + 1
    return float(n_variables / (n_variables - 1) * ((sum_h - h_joint) / sum_h))
    
