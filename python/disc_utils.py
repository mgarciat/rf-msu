"""Supervised MDL discretization utilities for numeric features.

This module implements Fayyad–Irani entropy-based threshold selection,
with optional Numba and joblib acceleration and a pandas convenience API.

Authors
-------
- Miguel Garcia Torres <mgarciat@upo.es>
- Thomas Vanhaeren

Affiliation
-----------
Data Science & Big Data Lab
Universidad Pablo de Olavide, Spain

License
-------
See the LICENSE file in the project repository.

"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    njit = None
    NUMBA_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    Parallel = None
    delayed = None
    JOBLIB_AVAILABLE = False


USE_NUMBA_TRANSFORM = NUMBA_AVAILABLE
# Joblib is supported and benchmarked, but not enabled globally by default:
# this data has thousands of columns but only dozens of rows, so thread
# scheduling can cost more than the work per feature. Pass use_joblib=True to
# FayyadIranisDiscretization when a larger data set benefits from it.
USE_JOBLIB_DISCRETIZATION = False
DEFAULT_N_JOBS = -1


# NEW: lightweight entropy helpers for the discretizer hot path.
# The old version called upo_lib.entropy_fromhistogram and
# upo_lib.conditional_entropy_onrows thousands of times on tiny arrays.
# These helpers keep the same formulas but avoid repeated np.clip and generic
# conversion overhead inside the recursive Fayyad-Irani scan.
def _xlogx_counts(values):
    """Evaluate ``x * ln(x)`` for nonnegative count values.

    Parameters
    ----------
    values : array_like
        Scalar or array of counts. Nonpositive values contribute zero.

    Returns
    -------
    numpy.ndarray
        Elementwise values using the convention ``0 * ln(0) = 0``."""
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values, dtype=float)
    mask = values > 0.0
    out[mask] = values[mask] * np.log(values[mask])
    return out


def _entropy_counts(counts):
    """Calculate Shannon entropy from unnormalized count frequencies.

    Parameters
    ----------
    counts : array_like
        Nonnegative category counts.

    Returns
    -------
    float
        Entropy in bits; zero if the total frequency is zero."""
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total == 0.0:
        return 0.0

    return float((-_xlogx_counts(counts).sum() + total * np.log(total)) / (total * np.log(2.0)))


def _conditional_entropy_count_tables(count_tables):
    """Calculate conditional entropy for one or more count tables.

    Parameters
    ----------
    count_tables : array_like
        Array of shape ``(..., n_rows, n_classes)``. Rows correspond to
        conditioning categories and columns to predicted categories.

    Returns
    -------
    numpy.ndarray or numpy.float64
        Conditional entropies in bits with shape ``(...)``. Zero-total
        tables have entropy zero."""
    count_tables = np.asarray(count_tables, dtype=float)
    row_sums = count_tables.sum(axis=-1)
    totals = row_sums.sum(axis=-1)
    numerators = -_xlogx_counts(count_tables).sum(axis=(-1, -2)) + _xlogx_counts(row_sums).sum(axis=-1)
    return np.divide(
        numerators,
        totals * np.log(2.0),
        out=np.zeros_like(numerators, dtype=float),
        where=totals > 0.0,
    )


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _transform_numba(X, cutpoint_matrix, cutpoint_lengths):
        n_samples, n_features = X.shape
        out = np.empty((n_samples, n_features), dtype=np.int64)

        for j in range(n_features):
            n_cuts = cutpoint_lengths[j]
            for i in range(n_samples):
                value = X[i, j]
                index = 0
                while index < n_cuts and value >= cutpoint_matrix[j, index]:
                    index += 1
                out[i, j] = index

        return out
else:
    _transform_numba = None


class MDLBasedDiscretizationModel:

    """Store learned cut points and apply them to unseen numeric observations.

    Parameters
    ----------
    cut_points : sequence of array_like
        Sorted cut points per input feature. The implementation does not
        sort or validate these values.

    Attributes
    ----------
    cut_points : list of numpy.ndarray
        Float thresholds for each feature."""
    def __init__(self, cut_points):
        """Initialize a discretization model from per-feature thresholds.

        Parameters
        ----------
        cut_points : sequence of array_like
            Sorted numeric cut points for each feature."""
        self.set_cut_points(cut_points)

    # -----------------------------
    # Getters / Setters
    # -----------------------------

    def get_cut_points(self):
        """Return the stored per-feature cut points.

        Returns
        -------
        list of numpy.ndarray
            Internal arrays of sorted cut points (not copies)."""
        return self.cut_points

    def set_cut_points(self, cut_points):
        # Aseguramos formato consistente
        """Replace the cut points and prepare the padded Numba representation.

        Parameters
        ----------
        cut_points : sequence of array_like
            Numeric threshold arrays, in ascending order for correct behavior.

        Notes
        -----
        The input ordering is retained without validation. Updates the internal
        padded matrix and lengths used by the optional compiled transform."""
        self.cut_points = [
            np.array(cp, dtype=float) for cp in cut_points
        ]
        self._cutpoint_lengths = np.array(
            [len(cp) for cp in self.cut_points],
            dtype=np.int64,
        )
        max_cuts = int(self._cutpoint_lengths.max()) if len(self._cutpoint_lengths) else 0
        self._cutpoint_matrix = np.empty((len(self.cut_points), max_cuts), dtype=np.float64)
        self._cutpoint_matrix.fill(np.inf)

        for j, cp in enumerate(self.cut_points):
            if len(cp) > 0:
                self._cutpoint_matrix[j, :len(cp)] = cp

    # -----------------------------
    # Representación
    # -----------------------------

    def __str__(self):
        """Summarize model dimensionality and learned cut points.

        Returns
        -------
        str
            Multi-line representation of thresholds for every feature."""
        lines = []
        n_features = len(self.cut_points)

        lines.append(
            f"Discretization model of {n_features} feature(s)."
        )

        for f, cps in enumerate(self.cut_points):
            lines.append(
                f"\tfeature #{f+1}/{n_features}; "
                f"\t#values {len(cps)} : {cps.tolist()}"
            )

        return "\n".join(lines)

    # -----------------------------
    # Discretización
    # -----------------------------

    def discretize(self, iarray):
        """Assign bin indices to a single observation.

        Parameters
        ----------
        iarray : array_like of shape (n_features,)
            Continuous feature values in the model feature order.

        Returns
        -------
        numpy.ndarray of int
            Bin indices with right-sided boundary convention: values equal to
            a cut point are assigned to the bin on its right."""

        iarray = np.asarray(iarray, dtype=float)
        oarray = np.zeros(len(iarray), dtype=int)

        for i, value in enumerate(iarray):

            cutpoints = self.cut_points[i]
            index = 0

            while index < len(cutpoints) and value >= cutpoints[index]:
                index += 1

            oarray[i] = index

        return oarray

    def transform(self, X, use_numba=None):
        """Discretize an observation matrix (or a single observation).

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features) or (n_features,)
            Continuous input observations.
        use_numba : bool or None, optional
            Enable the compiled implementation if available. ``None`` uses
            ``USE_NUMBA_TRANSFORM``. Otherwise NumPy is used.

        Returns
        -------
        numpy.ndarray of int
            Discretized observations of the same shape as the input.

        Raises
        ------
        ValueError
            If the input is not 1D/2D or a 2D input has an incompatible
            number of columns."""
        # OLD CODE at call sites:
        # X_discrete = np.array([mdl.model.discretize(row) for row in X])
        #
        # NEW CODE:
        # Transform a whole matrix column-by-column with np.searchsorted.
        # This preserves the same bin assignment as discretize(row), but avoids
        # Python loops over every row and every cut point.
        #
        # NEWER NUMBA PATH:
        # When Numba is installed and USE_NUMBA_TRANSFORM=True, use the
        # compiled _transform_numba loop over a padded cut-point matrix. This is
        # most useful after the first JIT compilation or in repeated runs.
        #
        # use_numba can override the module default for one call. Internal
        # training code uses use_numba=False because a one-shot transform is
        # faster with NumPy than paying Numba's first-call compile cost.
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return self.discretize(X)

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D o una instancia 1D.")

        if X.shape[1] != len(self.cut_points):
            raise ValueError(
                f"X debe tener {len(self.cut_points)} columnas, "
                f"pero tiene {X.shape[1]}."
            )

        use_numba = USE_NUMBA_TRANSFORM if use_numba is None else bool(use_numba)

        if use_numba and _transform_numba is not None:
            return _transform_numba(X, self._cutpoint_matrix, self._cutpoint_lengths)

        out = np.empty(X.shape, dtype=np.int64)
        for j, cutpoints in enumerate(self.cut_points):
            out[:, j] = np.searchsorted(cutpoints, X[:, j], side="right")
 
        return out

class MDLBasedDiscretization(ABC):

    """Abstract supervised discretizer that learns thresholds per feature.

    Parameters
    ----------
    data : array_like of shape (n_samples, n_features)
        Continuous predictors.
    labels : array_like of shape (n_samples,)
        Nonnegative integer class codes starting at zero.
    use_joblib : bool or None, optional
        Use joblib to process separate features concurrently, if present.
    n_jobs : int or None, optional
        Number of joblib workers. ``None`` uses ``DEFAULT_N_JOBS``."""
    def __init__(self, data, labels, *, use_joblib=None, n_jobs=None):
        """Construct the discretizer and immediately fit the cut-point model.

        Parameters
        ----------
        data : array_like of shape (n_samples, n_features)
            Continuous predictors.
        labels : array_like of shape (n_samples,)
            Nonnegative integer target labels.
        use_joblib : bool or None, optional
            Request feature-level parallelism if joblib is installed.
        n_jobs : int or None, optional
            Worker count; ``None`` selects the module default."""
        self.model = None
        self.num_classes = None
        self.use_joblib = (
            USE_JOBLIB_DISCRETIZATION
            if use_joblib is None
            else bool(use_joblib)
        )
        self.n_jobs = DEFAULT_N_JOBS if n_jobs is None else n_jobs
        self.build_discretization_model(data, labels)

    def build_discretization_model(self, data, labels):

        """Learn feature-wise cut points from labeled training observations.

        Parameters
        ----------
        data : array_like of shape (n_samples, n_features)
            Training attributes; each column is discretized separately.
        labels : array_like of shape (n_samples,)
            Target category codes, expected to be nonnegative integers.

        Notes
        -----
        The resulting fitted model is stored in ``self.model``. Training
        expects nonempty labels and matching sample counts."""
        data = np.asarray(data)
        labels = np.asarray(labels)

        self.num_classes = labels.max() + 1

        tdata = data.T

        # OLD CODE:
        # cutpoints = []
        # for feature in tdata:
        #     sorted_indices = np.argsort(feature)
        #     sfeature = feature[sorted_indices]
        #     slabels = labels[sorted_indices]
        #     cps = self.calculate_cut_points(sfeature, slabels, 0, len(sfeature))
        #     if cps is None:
        #         cps = [np.inf]
        #     cutpoints.append(cps)
        #
        # NEW JOBLIB PATH:
        # Feature discretization is independent by column, so when joblib is
        # available we parallelize columns with the threading backend. The
        # sequential helper remains the fallback and is still useful for tiny
        # inputs or environments without joblib.
        if self.use_joblib and JOBLIB_AVAILABLE and tdata.shape[0] > 1:
            cutpoints = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._calculate_feature_cut_points)(feature, labels)
                for feature in tdata
            )
        else:
            cutpoints = [
                self._calculate_feature_cut_points(feature, labels)
                for feature in tdata
            ]

        self.model = MDLBasedDiscretizationModel(cutpoints)

    def _calculate_feature_cut_points(self, feature, labels):
        """Determine cut points for one feature, after sorting training values.

        Parameters
        ----------
        feature : array_like of shape (n_samples,)
            Continuous values for a single predictor.
        labels : array_like of shape (n_samples,)
            Associated integer classes.

        Returns
        -------
        list of float
            Ordered thresholds; ``[numpy.inf]`` is used if no split qualifies."""
        sorted_indices = np.argsort(feature)
        sfeature = feature[sorted_indices]
        slabels = labels[sorted_indices]

        cps = self.calculate_cut_points(
            sfeature, slabels, 0, len(sfeature)
        )

        if cps is None:
            cps = [np.inf]

        return cps

    def calculate_cut_points(self, svalues, labels, first, last):

        """Recursively select entropy-reducing cut points accepted by ``split_test``.

        Parameters
        ----------
        svalues : numpy.ndarray
            Feature values sorted in nondecreasing order.
        labels : numpy.ndarray
            Integer class labels aligned with the sorted values.
        first : int
            Inclusive segment start position.
        last : int
            Exclusive segment end position.

        Returns
        -------
        list of float or None
            Sorted accepted cut points, or ``None`` for an unsplittable segment.

        Notes
        -----
        Evaluates all strictly increasing adjacent values as candidate splits;
        the split threshold is the midpoint of adjacent values."""
        if last - first < 2:
            return None

        # OLD CODE:
        # pcounts = np.zeros(self.num_classes)
        # for i in range(first, last):
        #     pcounts[labels[i]] += 1
        #
        # ccounts = np.zeros((2, self.num_classes))
        # ccounts[1] = pcounts.copy()
        # bcounts = np.zeros_like(ccounts)
        #
        # pentropy = pl.entropy_fromhistogram(pcounts)
        # bentropy = pentropy
        # best_index = -1
        # bcpoint = None
        # num_cutpoints = 0
        #
        # for i in range(first, last - 1):
        #     ccounts[0][labels[i]] += 1
        #     ccounts[1][labels[i]] -= 1
        #     if svalues[i] < svalues[i + 1]:
        #         cpoint = (svalues[i] + svalues[i + 1]) / 2.0
        #         centropy = pl.conditional_entropy_onrows(ccounts)
        #         if centropy < bentropy:
        #             bentropy = centropy
        #             bcpoint = cpoint
        #             best_index = i
        #             bcounts = ccounts.copy()
        #         num_cutpoints += 1
        #
        # NEW CODE:
        # Evaluate all valid candidate split points for this segment in one
        # vectorized pass. The best index, counts, gain, and MDL test are the
        # same quantities as before; only the mechanics are faster.
        segment_values = svalues[first:last]
        segment_labels = labels[first:last].astype(np.int64, copy=False)
        pcounts = np.bincount(segment_labels, minlength=self.num_classes).astype(float)
        pentropy = _entropy_counts(pcounts)

        candidate_offsets = np.flatnonzero(segment_values[:-1] < segment_values[1:])
        num_cutpoints = candidate_offsets.size
        if num_cutpoints == 0:
            return None

        one_hot = np.eye(self.num_classes, dtype=float)[segment_labels]
        left_counts = np.cumsum(one_hot, axis=0)[candidate_offsets]
        right_counts = pcounts - left_counts
        candidate_counts = np.stack((left_counts, right_counts), axis=1)

        entropies = _conditional_entropy_count_tables(candidate_counts)

        best_pos = int(np.argmin(entropies))
        bentropy = float(entropies[best_pos])
        best_index = int(first + candidate_offsets[best_pos])
        bcounts = candidate_counts[best_pos]
        bcpoint = (svalues[best_index] + svalues[best_index + 1]) / 2.0

        gain = float(pentropy - bentropy)
        num_instances = last - first

        if gain > 0 and self.split_test(
                pcounts, bcounts, num_instances, num_cutpoints):

            left = self.calculate_cut_points(
                svalues, labels, first, best_index + 1)

            right = self.calculate_cut_points(
                svalues, labels, best_index + 1, last)

            result = []

            if left is not None:
                result.extend(left)

            result.append(bcpoint)

            if right is not None:
                result.extend(right)

            return result

        return None

    @abstractmethod
    def split_test(self, prior_counts, best_counts,
                   num_instances, num_cutpoints):
        """Declare whether a candidate partition satisfies a stopping criterion.

        Parameters
        ----------
        prior_counts : numpy.ndarray
            Counts per class before splitting.
        best_counts : numpy.ndarray of shape (2, n_classes)
            Per-class counts on either side of the split.
        num_instances : int
            Number of samples in the evaluated segment.
        num_cutpoints : int
            Number of admissible candidate boundaries.

        Returns
        -------
        bool
            Whether the split is accepted by a concrete subclass."""
        pass


class FayyadIranisDiscretization(MDLBasedDiscretization):

    """Supervised discretization with the Fayyad–Irani MDL stopping rule.

    Notes
    -----
    Inherits feature-wise training and optional joblib parallelism from
    ``MDLBasedDiscretization``."""
    def split_test(self, prior_counts, best_counts,
                   num_instances, num_cutpoints):

        #prior_entropy = entropy(prior_counts)
        # OLD CODE:
        # prior_entropy = pl.entropy_fromhistogram(prior_counts)
        # entropy_after = pl.conditional_entropy_onrows(best_counts)
        # entropy_left = pl.entropy_fromhistogram(best_counts[0])
        # entropy_right = pl.entropy_fromhistogram(best_counts[1])
        #
        # NEW CODE:
        # Use the local count-array entropy helpers introduced above. Same
        # math, less overhead in a function called thousands of times.
        """Test the Fayyad–Irani minimum-description-length split criterion.

        Parameters
        ----------
        prior_counts : array_like
            Class frequencies before the split.
        best_counts : array_like of shape (2, n_classes)
            Class frequencies on left and right partitions.
        num_instances : int
            Number of observations in the segment.
        num_cutpoints : int
            Number of candidate thresholds considered in the segment.

        Returns
        -------
        bool
            ``True`` when information gain exceeds the MDL threshold.

        Notes
        -----
        Entropies are in bits. This implementation uses the number of
        candidate cut positions in the MDL threshold."""
        prior_entropy = _entropy_counts(prior_counts)
        #entropy_after = entropy_conditioned_on_rows(best_counts)
        entropy_after = float(_conditional_entropy_count_tables(best_counts))
        gain = prior_entropy - entropy_after

        num_classes_total = np.sum(prior_counts > 0)
        num_classes_left = np.sum(best_counts[0] > 0)
        num_classes_right = np.sum(best_counts[1] > 0)

        #entropy_left = entropy(best_counts[0])
        entropy_left = _entropy_counts(best_counts[0])
        #entropy_right = entropy(best_counts[1])
        entropy_right = _entropy_counts(best_counts[1])

        delta = (
            np.log2(3 ** num_classes_total - 2)
            - (
                num_classes_total * prior_entropy
                - num_classes_right * entropy_right
                - num_classes_left * entropy_left
            )
        )

        threshold = (np.log2(num_cutpoints) + delta) / num_instances

        return gain > threshold


def discretize(df: pd.DataFrame):
    """Fit MDL discretization and convert a pandas DataFrame to discrete bins.

    Parameters
    ----------
    df : pandas.DataFrame
        Training data with numeric predictors in every column except the
        last, which is treated as the categorical target.

    Returns
    -------
    ddf : pandas.DataFrame
        Discretized feature columns and integer-encoded target column.
    model : MDLBasedDiscretizationModel
        Fitted thresholds for applying the same transformation to new data.

    Notes
    -----
    The last column is converted to categorical codes via ``pandas.Categorical``.
    The cut points are fitted on the entire input; fit only on training data
    when evaluating predictive models to avoid data leakage."""
    # Separar X e y
    X = df.iloc[:, :-1].values
    y = pd.Categorical(df.iloc[:, -1]).codes

    cX = list(df.columns[:-1])
    cy = df.columns[-1]
    
    mdl = FayyadIranisDiscretization(X, y)
    
    # Reconstruir el DataFrame
    # OLD CODE:
    # X_discrete = np.array([mdl.model.discretize(row) for row in X])
    #
    # NEW CODE:
    # Use the matrix transform method, which is vectorized per feature.
    X_discrete = mdl.model.transform(X)
    
    ddf = pd.DataFrame(X_discrete, columns=cX)
    ddf[cy] = y

    return ddf, mdl.model
