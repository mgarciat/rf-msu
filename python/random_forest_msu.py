"""
Python port of Weka's weka.classifiers.trees.RandomForestMSU.

Design
------
Weka's RandomForestMSU extends RandomForest and replaces the base classifier
with RandomTreeMSU. This module follows the same design: it subclasses the
local Python RandomForest implementation and overrides only the pieces that
must create and fit TreeClassifierMSU instances.

Expected local modules
----------------------
- wekaRandomForestWM.py      -> contains RandomForest
- wekaRandomTreeMSUWM.py     -> contains TreeClassifierMSU
- wekaRandomTreeWM.py        -> base tree used by TreeClassifierMSU

Main extra options vs RandomForest
----------------------------------
- rediscretize_msu_per_node: passed to TreeClassifierMSU.fit
- X_msu_discrete: optional precomputed discrete matrix for MSU
- use_joblib_discretization / discretization_n_jobs: passed to TreeClassifierMSU
- bootstrap_mode:
    "sample"   -> physically trains each tree on the bootstrap sample.
                  This is usually safer for MSU if the MSU implementation is
                  unweighted, because zero-weight rows cannot influence MSU.
    "weighted" -> trains each tree on the full X with bootstrap counts as
                  weights, as in the previous RandomForestWM adaptation and
                  close to Weka's representCopiesUsingWeights=true.


MSU-based Random Forest classifier (Weka-inspired).

This module provides :class:`RandomForestMSU`, extending
:class:`random_forest.RandomForest` and using
:class:`random_tree_msu.TreeClassifierMSU` as its base learner.

The implementation supports a bootstrap sample mode and a weighted mode,
local or global MSU discretization, joblib forest parallelism, out-of-bag
evaluation, and inherited prediction/importance/summary methods.

**Compatibility note:** This variant uses the corrected TreeClassifierMSU from
random_tree_msu, based on random_tree, to fix the known
probability fallbacks without changing the forest aggregation algorithm.

Dependencies
------------
random_forest, random_tree_msu, random_tree, stat_utils, mstat_utils,
disc_utils, numpy, pandas; joblib and numba are optional accelerators.

Authors
-------
- Miguel Garcia Torres <mgarciat@upo.es>
- Thomas Vanhaeren

License
-------
See the project LICENSE file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from random_forest import RandomForest, JOBLIB_AVAILABLE, Parallel, delayed
from random_tree_msu import TreeClassifierMSU


@dataclass
class RandomForestMSU(RandomForest):
    """
    Port of ``weka.classifiers.trees.RandomForestMSU`` using
    ``TreeClassifierMSU`` as base tree.

    This class intentionally reuses the normal ``RandomForest`` implementation
    for prediction, probability alignment, OOB evaluation, average metrics,
    feature importance aggregation and scikit-learn-style parameter handling.

    The important difference is tree creation: every estimator is a
    ``TreeClassifierMSU`` instead of a normal ``TreeClassifier``.
    """

    # MSU-specific options forwarded to TreeClassifierMSU.fit
    rediscretize_msu_per_node: bool = True
    use_joblib_discretization: bool = False
    discretization_n_jobs: int = -1

    # See module docstring. "sample" avoids zero-weight rows affecting MSU.
    bootstrap_mode: str = "sample"

    # Learned MSU metadata, mainly useful when using global discretization.
    X_msu_discrete_: Optional[np.ndarray] = field(default=None, init=False)
    msu_encoding_: Optional[object] = field(default=None, init=False)

    # Narrower typing for clarity; RandomForest methods work with this list too.
    estimators_: List[TreeClassifierMSU] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate the inherited forest parameters and the MSU bootstrap mode.
        
        Raises
        ------
        ValueError
            If bootstrap_mode is not "sample" or "weighted".
        """
        super().__post_init__()
        if self.bootstrap_mode not in {"sample", "weighted"}:
            raise ValueError("bootstrap_mode debe ser 'sample' o 'weighted'.")

    def _make_tree(self, tree_seed: int) -> TreeClassifierMSU:
        """
        Create the MSU tree used by each forest iteration.

        Equivalent in spirit to the Java constructor of RandomForestMSU, which
        creates a RandomTreeMSU and installs it as the base classifier.
        """
        return TreeClassifierMSU(
            debug=self.debug,
            minNum=self.minNum,
            maxDepth=self.maxDepth,
            breakTiesRandomly=self.breakTiesRandomly,
            kValue=self.kValue,
            computeImpurityDecreasees=self.computeAttributeImportance,
            random_state=tree_seed,
            seed=tree_seed,
            use_joblib_discretization=self.use_joblib_discretization,
            discretization_n_jobs=self.discretization_n_jobs,
        )

    def _fit_msu_tree_from_bootstrap_spec(
        self,
        spec,
        X,
        y,
        base_weight,
        feature_types,
        feature_names,
        X_msu_discrete,
        rediscretize,
        use_joblib,
        n_jobs,
    ):
        """Fit one MSU tree from a precomputed bootstrap specification.
        
        This worker is used by the parallel execution path. It preserves the
        chosen sampling mode, discretization settings and bootstrap metadata.
        
        Returns
        -------
        tuple
            Tree index, fitted estimator, sampled indices, counts and seed.
        """
        i, indices, counts, tree_seed = spec
        tree = self._make_tree(tree_seed)

        if self.bootstrap_mode == "weighted":
            weights = base_weight * counts
            X_fit = X
            y_fit = y
            Xd_fit = X_msu_discrete
        else:
            X_fit = X[indices]
            y_fit = y[indices]
            weights = base_weight[indices].astype(np.float64, copy=True)
            Xd_fit = None if X_msu_discrete is None else X_msu_discrete[indices]

        tree.fit(
            X_fit,
            y_fit,
            feature_types=feature_types,
            feature_names=feature_names,
            sample_weight=weights,
            X_msu_discrete=Xd_fit,
            rediscretize_msu_per_node=rediscretize,
            use_joblib_discretization=use_joblib,
            discretization_n_jobs=n_jobs,
        )

        return i, tree, indices, counts, tree_seed

    def fit(
        self,
        X,
        y,
        feature_types=None,
        feature_names=None,
        sample_weight=None,
        X_msu_discrete: Optional[np.ndarray] = None,
        rediscretize_msu_per_node: Optional[bool] = None,
        use_joblib_discretization: Optional[bool] = None,
        discretization_n_jobs: Optional[int] = None,
    ):
        """
        Fit the RandomForestMSU model.

        Parameters are the same as RandomForest.fit, plus the MSU-specific
        parameters accepted by TreeClassifierMSU.fit.
        """
        X, y = self._validate_X_y(X, y)
        n_instances, n_features = X.shape
        self.n_features_in_ = n_features

        if sample_weight is None:
            base_weight = np.ones(n_instances, dtype=np.float64)
        else:
            base_weight = np.asarray(sample_weight, dtype=np.float64)
            if base_weight.ndim != 1:
                raise ValueError("sample_weight debe ser un vector 1D.")
            if base_weight.shape[0] != n_instances:
                raise ValueError("sample_weight debe tener la misma longitud que X e y.")

        if X_msu_discrete is not None:
            X_msu_discrete = np.asarray(X_msu_discrete, dtype=np.int64)
            if X_msu_discrete.shape != (n_instances, n_features):
                raise ValueError(
                    "X_msu_discrete debe tener shape "
                    f"({n_instances}, {n_features}), pero tiene {X_msu_discrete.shape}."
                )

        rediscretize = (
            self.rediscretize_msu_per_node
            if rediscretize_msu_per_node is None
            else rediscretize_msu_per_node
        )
        use_joblib = (
            self.use_joblib_discretization
            if use_joblib_discretization is None
            else use_joblib_discretization
        )
        n_jobs = (
            self.discretization_n_jobs
            if discretization_n_jobs is None
            else discretization_n_jobs
        )

        msu_encoding = None

        if (not rediscretize) and X_msu_discrete is None:
            # FOREST-LEVEL MSU DISCRETIZATION CACHE:
            # Old behavior: each bootstrap tree called TreeClassifierMSU.fit
            # with X_msu_discrete=None, so even "global" MSU mode relearned
            # discretization independently for every tree.
            #
            # New behavior: learn the global MSU discrete matrix once from the
            # forest training data, then pass row-aligned bootstrap slices to
            # each tree below. This keeps rediscretize_msu_per_node=False
            # meaningful at forest level and is the largest speed win here.
            prep_tree = self._make_tree(int(self._rng_seed()))
            X_msu_discrete, msu_encoding = prep_tree.prepare_msu_discrete(
                X,
                y,
                feature_types=feature_types,
                use_joblib=use_joblib,
                n_jobs=n_jobs,
            )

        self.classes_ = np.unique(y)
        self.feature_types = list(feature_types) if feature_types is not None else None
        self.feature_names = (
            list(feature_names) if feature_names is not None else [f"att{i}" for i in range(n_features)]
        )

        bag_size = int(round(n_instances * (self.bagSizePercent / 100.0)))
        bag_size = max(1, bag_size)

        rng = np.random.default_rng(self._rng_seed())
        self.estimators_ = []
        self.bootstrap_indices_ = []
        self.bootstrap_counts_ = []
        self.X_msu_discrete_ = X_msu_discrete
        self.msu_encoding_ = msu_encoding

        if self.debug:
            print("=== DEBUG RandomForestMSU.fit() ===")
            print(f"X.shape = {X.shape}")
            print(f"numIterations = {self.numIterations}")
            print(f"bagSizePercent = {self.bagSizePercent}")
            print(f"bag_size = {bag_size}")
            print(f"seed = {self._rng_seed()}")
            print(f"bootstrap_mode = {self.bootstrap_mode}")
            print(f"rediscretize_msu_per_node = {rediscretize}")

        if self.n_jobs != 1 and JOBLIB_AVAILABLE and self.numIterations > 1:
            # JOBLIB MSU FOREST PARALLELISM:
            # Bootstrap specs are generated once with the same RNG sequence as
            # the sequential loop. Each MSU tree fit is independent after that.
            # If X_msu_discrete was precomputed above, sample mode receives
            # row-aligned bootstrap slices and weighted mode receives the full
            # matrix, matching the sequential code below.
            specs = []
            for i in range(self.numIterations):
                indices = rng.integers(0, n_instances, size=bag_size, endpoint=False)
                counts = np.bincount(indices, minlength=n_instances).astype(np.float64)
                tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                specs.append((i, indices, counts, tree_seed))

            results = Parallel(
                n_jobs=self.n_jobs,
                backend=self.parallel_backend,
            )(
                delayed(self._fit_msu_tree_from_bootstrap_spec)(
                    spec,
                    X,
                    y,
                    base_weight,
                    feature_types,
                    feature_names,
                    X_msu_discrete,
                    rediscretize,
                    use_joblib,
                    n_jobs,
                )
                for spec in specs
            )

            for i, tree, indices, counts, tree_seed in sorted(results, key=lambda item: item[0]):
                if self.msu_encoding_ is None and getattr(tree, "msu_encoding_", None) is not None:
                    self.msu_encoding_ = tree.msu_encoding_

                self.estimators_.append(tree)
                self.bootstrap_indices_.append(indices)
                self.bootstrap_counts_.append(counts)

                if self.debug:
                    print(
                        f"Tree {i + 1}/{self.numIterations}: "
                        f"seed={tree_seed}, oob={(counts == 0).sum()}"
                    )

            if self.calcOutOfBag:
                self._compute_oob_score(X, y)

            return self

        for i in range(self.numIterations):
            indices = rng.integers(0, n_instances, size=bag_size, endpoint=False)
            counts = np.bincount(indices, minlength=n_instances).astype(np.float64)
            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            tree = self._make_tree(tree_seed)

            if self.bootstrap_mode == "weighted":
                # Same strategy as the previous RandomForestWM adaptation:
                # keep full X and represent bootstrap copies with weights.
                # Caveat: this is only fully equivalent if the MSU score also
                # respects zero/positive sample weights.
                weights = base_weight * counts
                X_fit = X
                y_fit = y
                Xd_fit = X_msu_discrete
            else:
                # Safer for unweighted MSU: the MSU matrix only contains rows
                # that are actually in the bootstrap sample. Duplicates are kept.
                X_fit = X[indices]
                y_fit = y[indices]
                weights = base_weight[indices].astype(np.float64, copy=True)
                Xd_fit = None if X_msu_discrete is None else X_msu_discrete[indices]

            tree.fit(
                X_fit,
                y_fit,
                feature_types=feature_types,
                feature_names=feature_names,
                sample_weight=weights,
                X_msu_discrete=Xd_fit,
                rediscretize_msu_per_node=rediscretize,
                use_joblib_discretization=use_joblib,
                discretization_n_jobs=n_jobs,
            )

            # Keep first non-empty encoding for inspection/debugging.
            if self.msu_encoding_ is None and getattr(tree, "msu_encoding_", None) is not None:
                self.msu_encoding_ = tree.msu_encoding_

            self.estimators_.append(tree)
            self.bootstrap_indices_.append(indices)
            self.bootstrap_counts_.append(counts)

            if self.debug:
                print(
                    f"Tree {i + 1}/{self.numIterations}: "
                    f"seed={tree_seed}, oob={(counts == 0).sum()}"
                )

        if self.calcOutOfBag:
            self._compute_oob_score(X, y)

        return self

    def get_params(self, deep=True):
        """Return inherited and MSU-specific hyperparameters as a dictionary.
        """
        params = super().get_params(deep=deep)
        params.update(
            {
                "rediscretize_msu_per_node": self.rediscretize_msu_per_node,
                "use_joblib_discretization": self.use_joblib_discretization,
                "discretization_n_jobs": self.discretization_n_jobs,
                "bootstrap_mode": self.bootstrap_mode,
            }
        )
        return params

    def set_params(self, **params):
        """Assign supplied hyperparameters and revalidate constructor settings.
        
        Returns
        -------
        RandomForestMSU
            Updated estimator.
        """
        valid = self.get_params().keys()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Parámetro desconocido: {key}")
            setattr(self, key, value)
        self.__post_init__()
        return self

    def to_string(self, print_classifiers=False, num_decimal_places=2):
        """Return a human-readable forest summary with MSU configuration.
        """
        if not self.estimators_:
            return "RandomForestMSU: No model built yet."

        text = super().to_string(
            print_classifiers=print_classifiers,
            num_decimal_places=num_decimal_places,
        )
        header = [
            "RandomForestMSU",
            "",
            f"Base classifier: TreeClassifierMSU",
            f"Bootstrap mode: {self.bootstrap_mode}",
            f"Rediscretize MSU per node: {self.rediscretize_msu_per_node}",
            f"Forest-level X_msu_discrete cache: {self.X_msu_discrete_ is not None}",
            f"n_jobs: {self.n_jobs}",
            f"parallel_backend: {self.parallel_backend}",
        ]

        # Replace the first line produced by RandomForest.to_string().
        parts = text.splitlines()
        if parts and parts[0] == "RandomForest":
            parts = parts[1:]
        return "\n".join(header + parts)

    def __str__(self):
        """Return the forest summary text.
        """
        return self.to_string(print_classifiers=False)


# Explicit aliases for common styles.
RandomForestMSUClassifier = RandomForestMSU
RandomForestClassifierMSU = RandomForestMSU
