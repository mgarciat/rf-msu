"""Random Tree with Multivariate Symmetrical Uncertainty (MSU).

This module extends :mod:`random_tree` rather than replacing its classes.
The class :class:`TreeMSU` inherits from ``Tree`` and
:class:`TreeClassifierMSU` inherits from ``TreeClassifier``.

The auxiliary discretized matrix is used for MSU attribute selection; the
original observations are used to compute actual categorical/numeric splits.
Local discretization recalculates thresholds at each node, while global
discretization propagates one shared integer-coded matrix down the tree.

Examples
--------
>>> import numpy as np
>>> X = np.array([["a"], ["a"], ["b"], ["b"]], dtype=object)
>>> y = np.array([0, 0, 1, 1])
>>> clf = TreeClassifierMSU(debug=False, kValue=1)
>>> clf.fit(X, y, feature_types=["categorical"])
>>> clf.predict(X).tolist()
[0, 0, 1, 1]

Notes
-----
Requires NumPy, pandas, ``mstat_utils``, ``disc_utils`` and ``random_tree``.
The MSU calculation uses integer-coded discrete variables and is based on
unweighted empirical frequencies; training weights affect split distributions
but are not explicitly passed to ``msuXy``.

Python port of Weka's TreeClassifierMSU idea, built on top of random_tree.Tree and random_tree.TreeClassifier.

Main classes
------------
TreeMSU
    Extends Tree. It keeps the original TreeClassifier split machinery, but selects
    split attributes with Multivariate Symmetrical Uncertainty (MSU).

RandomTreeMSU
    Extends TreeClassifier. It creates a TreeMSU root and prepares the discrete
    matrix used by MSU.

Notes
-----
- X is used for the actual tree split, exactly as in the base Tree.
- X_msu_discrete is used only to compute MSU. It must contain integer-coded
  discrete variables, one column per original feature.
- If X_msu_discrete is not supplied, a conservative ordinal encoding is built
  automatically. For closer parity with Weka's ClassificationDatasetAdapter,
  pass the same discrete representation explicitly.

Authors
-------
- Miguel Garcia Torres <mgarciat@upo.es>
- Thomas Vanhaeren

License
-------
See the project LICENSE file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List
import math
import random as py_random

import numpy as np
import pandas as pd

import sys

import mstat_utils as uml
import disc_utils as ud
from random_tree import Tree, TreeClassifier


@dataclass
class TreeMSU(Tree):
    """
    Tree node that extends the base Tree with MSU-based attribute selection.

    The inherited methods `distribution`, `split_data`, `to_string`,
    `summary_string`, etc. are reused. This class adds:
      - MSU subset propagation across recursive calls.
      - Splitting of the auxiliary discrete MSU matrix in sync with X/y/weights.
    """


    WEKA_SMALL: float = 1e-6

    @staticmethod
    def weka_gr(a: float, b: float) -> bool:
        """Approximation of Weka Utils.gr(a, b)."""
        return (a - b) > TreeMSU.WEKA_SMALL

    @staticmethod
    def weka_eq(a: float, b: float) -> bool:
        """Approximation of Weka Utils.eq(a, b)."""
        return abs(a - b) <= TreeMSU.WEKA_SMALL


    def _discretize_msu_for_node(
            self,
            X: np.ndarray,
            y_int: np.ndarray,
    ) -> np.ndarray:
        """
        Construye X_msu_discrete para el nodo actual.
        
        Equivale a ClassificationDatasetAdapter.newInstance(data) en Java:
        toma sólo las instancias del nodo y vuelve a discretizar los atributos
        numéricos usando Fayyad-Irani sobre ese subconjunto.
        """
        X = np.asarray(X, dtype=object)
        y_int = np.asarray(y_int, dtype=np.int64)

        n_samples, n_features = X.shape
        Xd = np.zeros((n_samples, n_features), dtype=np.int64)

        numeric_indices = []
        
        for j in range(n_features):
            
            if self.feature_types[j] == "categorical":
                col = X[:, j]
                
                values = []
                for v in col:
                    if self.is_missing_value(v):
                        continue
                    if v not in values:
                        values.append(v)

                mapping = {v: i for i, v in enumerate(values)}
                missing_code = len(mapping)

                for i, v in enumerate(col):
                    if self.is_missing_value(v):
                        Xd[i, j] = missing_code
                    else:
                        Xd[i, j] = mapping[v]

            elif self.feature_types[j] in ("numeric", "real"):
                numeric_indices.append(j)

            else:
                raise ValueError(
                    f"Tipo de atributo desconocido en columna {j}: "
                    f"{self.feature_types[j]}"
                )

        if len(numeric_indices) > 0:
            X_num = np.asarray(X[:, numeric_indices], dtype=float)

            disc = ud.FayyadIranisDiscretization(X_num, y_int)
            numeric_model = disc.model
            X_num_discrete = numeric_model.transform(X_num)

            for pos, j in enumerate(numeric_indices):
                Xd[:, j] = X_num_discrete[:, pos]

        return Xd


    
    def distribution(self, props, dists, att, X, y, weights):
        """
        Weighted classification distribution for a candidate split attribute.

        This is the weighted version of the base Tree.distribution(). It is
        included here so TreeMSU works even when the imported base Tree still has
        the older unweighted signature.
        """
        if self.feature_types is None:
            raise ValueError("self.feature_types no puede ser None.")

        if self.n_classes is None:
            raise ValueError("self.n_classes no puede ser None. Debe fijarse desde RandomTreeMSU.")

        split_point = np.nan
        self.last_sorted_indices = None
        self.last_sorted_att = -1
        
        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)

        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights debe tener la misma longitud que X e y.")

        n_instances = X.shape[0]
        n_classes = self.n_classes
        attribute_type = self.feature_types[att]

        if attribute_type == "categorical":
            if self.categorical_values is None or att not in self.categorical_values:
                raise ValueError(f"No hay categorical_values para el atributo categórico {att}.")

            values = self.categorical_values[att]
            n_values = len(values)
            value_to_index = {v: i for i, v in enumerate(values)}
            dist = np.zeros((n_values, n_classes), dtype=float)

            for i in range(n_instances):
                att_val = X[i, att]

                if self.is_missing_value(att_val):
                    continue

                if att_val not in value_to_index:
                    raise ValueError(f"Valor categórico desconocido en atributo {att}: {att_val}")

                row = value_to_index[att_val]
                cls = int(y[i])
                dist[row, cls] += weights[i]

        elif attribute_type in ("numeric", "real"):
            curr_dist = np.zeros((2, n_classes), dtype=float)
            dist = np.zeros((2, n_classes), dtype=float)

            col = X[:, att].astype(float)
            non_missing_mask = ~np.isnan(col)
            non_missing_indices = np.where(non_missing_mask)[0]
            missing_indices = np.where(~non_missing_mask)[0]
            sorted_non_missing_indices = non_missing_indices[np.argsort(col[non_missing_indices])]
            sorted_indices = np.concatenate([sorted_non_missing_indices, missing_indices])

            self.last_sorted_indices = sorted_indices.copy()
            self.last_sorted_att = att
            
            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]
            weights_sorted = weights[sorted_indices]
            col_sorted = col[sorted_indices]
            index_of_first_missing_value = len(sorted_non_missing_indices)

            for j in range(index_of_first_missing_value):
                cls = int(y_sorted[j])
                curr_dist[1, cls] += weights_sorted[j]

            prior_val = self.prior_val(curr_dist)
            dist[:, :] = curr_dist

            curr_split = col_sorted[0] if index_of_first_missing_value > 0 else np.nan
            best_val = -np.inf

            for i in range(index_of_first_missing_value):
                att_val = col_sorted[i]

                if att_val > curr_split:
                    curr_val = self.gain(curr_dist, prior_val)

                    if curr_val > best_val:
                        best_val = curr_val
                        split_point = (att_val + curr_split) / 2.0
                        if split_point <= curr_split:
                            split_point = att_val
                        dist[:, :] = curr_dist

                    curr_split = att_val

                cls = int(y_sorted[i])
                w = weights_sorted[i]
                curr_dist[0, cls] += w
                curr_dist[1, cls] -= w

            X = X_sorted
            y = y_sorted
            weights = weights_sorted

        else:
            raise ValueError(f"Tipo de atributo desconocido: {attribute_type}")

        props[0] = np.sum(dist, axis=1).astype(float)
        total_props = float(np.sum(props[0]))
        if np.isclose(total_props, 0.0):
            props[0][:] = 1.0 / len(props[0])
        else:
            props[0] = props[0] / total_props

        for i in range(n_instances):
            att_val = X[i, att]
            if self.is_missing_value(att_val):
                cls = int(y[i])
                w = weights[i]
                for j in range(dist.shape[0]):
                    dist[j, cls] += w * props[0][j]

        dists[0] = dist
        return split_point

    msu_subset: Optional[np.ndarray] = None

    def _extend_msu_subset(self, msu_subset: Optional[np.ndarray]) -> np.ndarray:
        """
        Equivalent to the Java helper extendMsuSubset(int[] msuSubset):
        append one free slot where the candidate attribute will be written.
        """
        if msu_subset is None:
            return np.zeros(1, dtype=np.int64)

        msu_subset = np.asarray(msu_subset, dtype=np.int64)
        extended = np.zeros(msu_subset.size + 1, dtype=np.int64)
        extended[:-1] = msu_subset
        return extended

    def _msu_score(
        self,
        X_msu_discrete: np.ndarray,
        y: np.ndarray,
        trial_subset: np.ndarray,
    ) -> float:
        """Compute MSU(X_trial_subset, y)."""
        trial_subset = np.asarray(trial_subset, dtype=np.int64)

        #print("###########X_msu")
        #print(X_msu_discrete)
        #print("###########y")
        #print(y)
        #print("###########MSU subset")
        #print(trial_subset)
        
        if trial_subset.size == 0:
            return 0.0

        if np.any(trial_subset < 0) or np.any(trial_subset >= X_msu_discrete.shape[1]):
            raise ValueError(
                f"msu_subset contiene índices fuera de rango: {trial_subset.tolist()}"
            )

        return float(uml.msuXy(X_msu_discrete, y, trial_subset))

    def split_data_msu(
        self,
        X: np.ndarray,
        X_msu_discrete: Optional[np.ndarray],
        y: np.ndarray,
        weights: np.ndarray,
        row_ids: np.ndarray,
    ) -> list[tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split X, X_msu_discrete, y and weights using the selected split.

        This mirrors Tree.split_data(), but keeps the MSU discrete matrix aligned
        with duplicated/reweighted missing-value instances.

        Returns
        -------
        list of tuples:
            [(X0, Xd0, y0, w0, ids0), (X1, Xd1, y1, w1, ids1), ...]
        """
        # OLD CODE:
        # split_data_msu only split X, y, weights, and row_ids. That meant the
        # discrete MSU matrix was lost at the first child.
        #
        # NEW CODE:
        # X_msu_discrete is optional. When present, it is duplicated/reweighted
        # alongside X for missing values and returned per child.
        X = np.asarray(X, dtype=object)
        X_msu_discrete = (
            None
            if X_msu_discrete is None
            else np.asarray(X_msu_discrete, dtype=np.int64)
        )
        y = np.asarray(y)
        weights = np.asarray(weights, dtype=np.float64)
        
        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights debe tener la misma longitud que X e y.")

        if X_msu_discrete is not None and X_msu_discrete.shape[0] != X.shape[0]:
            raise ValueError("X_msu_discrete debe tener la misma longitud que X.")

        if self.attribute < 0:
            raise ValueError("No se puede dividir: self.attribute == -1")

        if self.proportion is None:
            raise ValueError("No se puede dividir: self.proportion is None")

        n_branches = len(self.proportion)
        X_subsets = [[] for _ in range(n_branches)]
        Xd_subsets = [[] for _ in range(n_branches)]
        y_subsets = [[] for _ in range(n_branches)]
        w_subsets = [[] for _ in range(n_branches)]
        id_subsets = [[] for _ in range(n_branches)]
        
        attribute_type = self.feature_types[self.attribute]

        if attribute_type == "categorical":
            values = self.categorical_values[self.attribute]
            value_to_index = {v: i for i, v in enumerate(values)}

        for i in range(X.shape[0]):
            value = X[i, self.attribute]

            if self.is_missing_value(value):
                for k in range(n_branches):
                    if self.proportion[k] > 0:
                        X_subsets[k].append(X[i].copy())
                        if X_msu_discrete is not None:
                            Xd_subsets[k].append(X_msu_discrete[i].copy())
                        y_subsets[k].append(y[i])
                        w_subsets[k].append(weights[i] * self.proportion[k])
                        id_subsets[k].append(row_ids[i])
                continue

            if attribute_type == "categorical":
                if value not in value_to_index:
                    raise ValueError(
                        f"Valor categórico desconocido en atributo {self.attribute}: {value}"
                    )
                branch = value_to_index[value]

            elif attribute_type in ("numeric", "real"):
                branch = 0 if float(value) < self.splitPoint else 1

            else:
                raise ValueError(f"Tipo de atributo desconocido: {attribute_type}")

            X_subsets[branch].append(X[i].copy())
            if X_msu_discrete is not None:
                Xd_subsets[branch].append(X_msu_discrete[i].copy())
            y_subsets[branch].append(y[i])
            w_subsets[branch].append(weights[i])
            id_subsets[branch].append(row_ids[i])
        subsets = []
        for k in range(n_branches):
            if len(X_subsets[k]) == 0:
                Xk = np.empty((0, X.shape[1]), dtype=object)
                Xdk = (
                    None
                    if X_msu_discrete is None
                    else np.empty((0, X_msu_discrete.shape[1]), dtype=np.int64)
                )
                yk = np.empty((0,), dtype=y.dtype)
                wk = np.empty((0,), dtype=np.float64)
                idk = np.empty((0,), dtype=np.int64)
            else:
                Xk = np.asarray(X_subsets[k], dtype=object)
                Xdk = (
                    None
                    if X_msu_discrete is None
                    else np.asarray(Xd_subsets[k], dtype=np.int64)
                )
                yk = np.asarray(y_subsets[k], dtype=y.dtype)
                wk = np.asarray(w_subsets[k], dtype=np.float64)
                idk = np.asarray(id_subsets[k], dtype=np.int64)


            subsets.append((Xk, Xdk, yk, wk, idk))

        return subsets

    
    def build_tree_msu(
        self,
        X: np.ndarray,
        X_msu_discrete: Optional[np.ndarray],
        y: np.ndarray,
        weights: np.ndarray,
        class_probs: np.ndarray,
        att_indices_window: np.ndarray,
        total_weight: float,
        random: Any,
        depth: int,
        msu_subset: Optional[np.ndarray] = None,
        debug_state = None,
        row_ids = None,
    ) -> None:
        """Recursively grow a tree using MSU-based attribute selection.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Original predictor values used for tree splitting.
        X_msu_discrete : ndarray of int or None
            Auxiliary discrete matrix. If ``None``, rediscretize each node.
        y : ndarray of int
            Class indices local to the trained classifier.
        weights : ndarray of float
            Observation weights used for split distributions.
        class_probs : ndarray of float
            Weighted class counts at the current node.
        att_indices_window : ndarray of int
            Candidate feature indices, reordered during attribute sampling.
        total_weight : float
            Nominal node weight.
        random : random.Random
            Shared pseudorandom generator.
        depth : int
            Current tree depth.
        msu_subset : ndarray of int or None, optional
            Attribute indices used along the current path, potentially repeated.
        debug_state : dict or None, optional
            Shared candidate-evaluation counter.
        row_ids : ndarray of int or None, optional
            Indices of original observations tracked through recursive splits.

        Returns
        -------
        None
            Updates the node in place and creates child ``TreeMSU`` objects.

        Notes
        -----
        MSU is computed on ``X_msu_discrete`` and is not explicitly weighted;
        split distributions and propagated training samples are weighted.
        A candidate producing an empty branch is skipped.
        """
        if debug_state is None:
            debug_state = {"global_it": 0}

        if row_ids is None:
            row_ids = np.arange(X.shape[0], dtype=np.int64)

        row_ids = np.asarray(row_ids, dtype=np.int64)

        if row_ids.shape[0] != X.shape[0]:
            raise ValueError("row_ids debe tener la misma longitud que X.")
        
        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)
        class_probs = np.asarray(class_probs, dtype=np.float64)
        att_indices_window = np.asarray(att_indices_window, dtype=np.int64)

        self.weights = weights
        self.msu_subset = None if msu_subset is None else np.asarray(msu_subset, dtype=np.int64)
        
        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights debe tener la misma longitud que X e y.")

        ninstances, nfeatures = X.shape

        if ninstances == 0:
            self.attribute = -1
            self.classDistribution = None
            self.proportion = None
            self.treeNodes = None
            return

        # Preserve weighted class counts at all nonempty internal MSU nodes.
        # Required by random_tree.Tree.distribution_for_instance fallback
        # when the selected categorical value was not seen at this node.
        self.classDistribution = class_probs.copy()

        total_weight = float(np.sum(class_probs))

        is_pure_classification_node = np.isclose(
            class_probs[np.argmax(class_probs)],
            total_weight,
        )

        max_depth_reached = self.maxDepth > 0 and depth >= self.maxDepth

        if self.debug:
            print("==========build tree MSU==========")
            print(f"total_weight = {total_weight}")
            print(f"minNum = {self.minNum}")
            print(f"is_pure_classification_node = {is_pure_classification_node}")
            print(f"max_depth_reached = {max_depth_reached}")
            print(f"depth = {depth}, maxDepth = {self.maxDepth}")
            print(f"msu_subset = {None if msu_subset is None else list(msu_subset)}")
            print(f"#instancias: {len(row_ids)}")
            print("row_ids =", row_ids.tolist())
        if (
            total_weight < 2 * self.minNum
            or is_pure_classification_node
            or max_depth_reached
        ):
            if self.debug:
                print("  L]Creating leaf node")

            self.attribute = -1
            self.classDistribution = class_probs.copy()
            self.proportion = None
            self.treeNodes = None
            return

        #X_msu_node = self._discretize_msu_for_node(X, y)
        # OLD CODE:
        # Child nodes were always called with X_msu_discrete=None, which forced
        # rediscretization even when fit(..., rediscretize_msu_per_node=False)
        # requested one global discretization.
        #
        # NEW CODE:
        # If a discrete matrix is supplied, keep it aligned through the split
        # and pass child matrices down the recursion.
        reuse_discretization = X_msu_discrete is not None

        if X_msu_discrete is None:
            X_msu_node = self._discretize_msu_for_node(X, y)
        else:
            X_msu_node = np.asarray(X_msu_discrete, dtype=np.int64)

        
        val = -np.inf
        split = np.nan
        best_dists = None
        best_props = None
        best_index = 0
        best_msu_subset = None

        props = [None]
        dists = [None]

        window_size = nfeatures
        k = self.kValue
        gain_found = False
        
        base_subset = (
            np.array([], dtype=np.int64)
            if msu_subset is None
            else np.asarray(msu_subset, dtype=np.int64).copy()
        )

        best_msu_subset = base_subset.copy()

        if self.debug:
            print(f"att_indices_window = {att_indices_window}")
            print(f"initial base MSU subset = {base_subset}")

        best_sorted_indices = None
            
        while window_size > 0 and ((k > 0) or (not gain_found)):
            k -= 1

            chosen_index = random.randrange(window_size)
            att_index = int(att_indices_window[chosen_index])

            global_it = debug_state["global_it"]
            debug_state["global_it"] += 1
            
            if self.debug:
                print("\t\n--- while attributes MSU ---")
                print(f"\t window_size = {window_size}")
                print(f"\t k after decrement = {k}")
                print(f"\t chosen_index = {chosen_index}")
                print(f"\t att_index = {att_index}")
                
            att_indices_window[chosen_index] = att_indices_window[window_size - 1]
            att_indices_window[window_size - 1] = att_index
            window_size -= 1

            curr_split = self.distribution(props, dists, att_index, X, y, weights)

            attribute_type = self.feature_types[att_index]

            if attribute_type in ("numeric", "real") and np.isnan(curr_split):
                if self.debug:
                    print(f"\t skipping attribute {att_index} because curr_split is NaN")
                continue

            if props[0] is None or np.any(np.isclose(props[0], 0.0)):
                if self.debug:
                    print(f"\t skipping attribute {att_index} because it creates an empty branch: props={props[0]}")
                continue

            
            # Weka JRandomTreeMSU permite repetir atributos en el subconjunto MSU.
            # Por tanto, aunque att_index ya esté en base_subset, se vuelve a añadir.
            trial_subset = np.append(base_subset, att_index).astype(np.int64)

            debug_b = False

            if debug_b:
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("\n" + "=" * 80)
                print(f"DEBUG TARGET ITERATION {global_it}")
                print("=" * 80)
                
                print(f"depth = {depth}")
                print(f"n_instances = {X.shape[0]}")
                print(f"total_weight = {np.sum(weights)}")
                print(f"class_probs = {class_probs}")
                print(f"att_indices_window BEFORE swap? revisar impresión previa")
                print(f"window_size = {window_size}")
                print(f"k = {k}")
                print(f"chosen_index = {chosen_index}")
                print(f"att_index = {att_index}")
                print(f"base_subset = {base_subset}")
                print(f"trial_subset = {trial_subset}")
            
#            print("y =", y)
#            print("weights =", weights)

#            print("X[:, 0] =", X[:, 0])
#            print("X[:, 1] =", X[:, 1])
#            print("X[:, 2] =", X[:, 2])
#            print("X[:, 3] =", X[:, 3])
                
#            print("X_msu_discrete[:, 0] =", X_msu_discrete[:, 0])
#            print("X_msu_discrete[:, 1] =", X_msu_discrete[:, 1])
#            print("X_msu_discrete[:, 2] =", X_msu_discrete[:, 2])
#            print("X_msu_discrete[:, 3] =", X_msu_discrete[:, 3])

#            print("X_msu_discrete[:, trial_subset] =")
#            print(X_msu_discrete[:, trial_subset])
                
#            print("unique y =", np.unique(y, return_counts=True))
#            print("unique weights =", np.unique(weights, return_counts=True))


#            for a in np.unique(trial_subset):
#                print(
#                    f"unique Xd[:, {a}] =",
#                    np.unique(X_msu_node[:, a], return_counts=True)
#                )
            
#            print("y")
#            print(y)
#            print("MSU subset (trial_subset):")
#            print(trial_subset)
            
            curr_val = self._msu_score(X_msu_node, y, trial_subset)

            #if global_it == 3:
            #    print("X")
            #    print(X)
            #    print("X_msu")
            #    print(X_msu_node)
            #    print(f"curr_msu_val = {curr_val}")
            if debug_b:
                print(f"curr_split = {curr_split}")
                print(f"props = {props[0]}")
                print(f"dists =\n{dists[0]}")
                print(f"curr_msu_val = {curr_val}")
                print("=" * 80 + "\n")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            #if global_it == 11:
            #    sys.exit(-1)


            
            if self.debug:
                print(f"\t curr_split = {curr_split}")
                print(f"\t MSU trial_subset = {trial_subset}")
                print(f"\t curr_msu_val = {curr_val}")

                # En Weka, gainFound se activa con el valor actual.
            if self.weka_gr(curr_val, 0.0):
                gain_found = True
                if self.debug:
                    print("\t gain_found set to True")

            # Comparación estilo Weka.
            if (
                self.weka_gr(curr_val, val)
                or (
                    not self.breakTiesRandomly
                    and self.weka_eq(curr_val, val)
                and att_index < best_index
                )
            ):
                val = curr_val
                best_index = att_index
                split = curr_split
                best_props = props[0].copy()
                best_dists = dists[0].copy()
                best_msu_subset = trial_subset.copy()
                
                if self.last_sorted_att == att_index and self.last_sorted_indices is not None:
                    best_sorted_indices = self.last_sorted_indices.copy()
                else:
                    best_sorted_indices = None
                    
                if self.debug:
                    print("\t Best MSU attribute updated")
                    print(f"\t val = {val}")
                    print(f"\t best_index = {best_index}")
                    print(f"\t split = {split}")
                    print(f"\t best_msu_subset = {best_msu_subset}")
                
                self.attribute = best_index

        if self.debug:
            print(f"best_index = {best_index}")
            print(f"best MSU val = {val}")

        if self.weka_gr(val, 0.0):
            if self.computeImpurityDecreasees and self.impurityDecreasees is not None:
                self.impurityDecreasees[self.attribute][0] += val
                self.impurityDecreasees[self.attribute][1] += 1

            self.splitPoint = split
            self.proportion = best_props
            self.msu_subset = best_msu_subset

            if best_sorted_indices is not None:
                X_for_split = X[best_sorted_indices]
                Xd_for_split = X_msu_node[best_sorted_indices] if reuse_discretization else None
                y_for_split = y[best_sorted_indices]
                weights_for_split = weights[best_sorted_indices]
                row_ids_for_split = row_ids[best_sorted_indices]
            else:
                X_for_split = X
                Xd_for_split = X_msu_node if reuse_discretization else None
                y_for_split = y
                weights_for_split = weights
                row_ids_for_split = row_ids
            
            # OLD CODE:
            # subsets = self.split_data_msu(
            #     X_for_split, y_for_split, weights_for_split, row_ids_for_split
            # )
            #
            # NEW CODE:
            # Split X_msu_discrete in the same way as X/y/weights so child
            # nodes can reuse the global discretization.
            subsets = self.split_data_msu(
                X_for_split,
                Xd_for_split,
                y_for_split,
                weights_for_split,
                row_ids_for_split,
            )
            self.treeNodes = [None] * len(best_dists)

            if self.debug:
                print(f"self.splitPoint = {self.splitPoint}")
                print(f"self.proportion = {self.proportion}")
                print(f"len(subsets) = {len(subsets)}")
                print(f"len(best_dists) = {len(best_dists)}")
            
                
            for i in range(len(best_dists)):

                X_child, Xd_child, y_child, w_child, ids_child = subsets[i]
                
                child = TreeMSU(
                    debug=self.debug,
                    minNum=self.minNum,
                    maxDepth=self.maxDepth,
                    breakTiesRandomly=self.breakTiesRandomly,
                    kValue=self.kValue,
                    computeImpurityDecreasees=self.computeImpurityDecreasees,
                    random_state=self.random_state,
                    seed=self.seed,
                    weights=w_child,
                    feature_types=self.feature_types,
                    categorical_values=self.categorical_values,
                    n_classes=self.n_classes,
                    msu_subset=best_msu_subset,
                )
                child.impurityDecreasees = self.impurityDecreasees

                child_total_weight = float(np.sum(w_child))

                if self.debug:
                    print(f"Building MSU child {i}, child_total_weight = {child_total_weight}")
                
                    
                child_class_probs = np.bincount(
                    y_child.astype(np.int64),
                    weights=w_child,
                    minlength=self.n_classes,
                ).astype(np.float64)

                if self.debug:
                    print(f"Building MSU child {i}, child_total_weight = {child_total_weight}")
                    print(f"child_class_probs recomputed = {child_class_probs}")
                    print(f"child y bincount = {np.bincount(y_child.astype(np.int64), minlength=self.n_classes)}")
                    print(f"child weights by class = {child_class_probs}")
                    
                child.build_tree_msu(
                    X_child,
                    Xd_child if reuse_discretization else None,
                    y_child,
                    w_child,
                    child_class_probs, #best_dists[i],
                    att_indices_window,
                    child_total_weight,
                    random,
                    depth + 1,
                    best_msu_subset,
                    debug_state,
                    ids_child,
                )

                self.treeNodes[i] = child

            empty_successor = any(subset[0].shape[0] == 0 for subset in subsets)

            if self.debug:
                print(f"empty_successor = {empty_successor}")

            if empty_successor:
                self.classDistribution = class_probs.copy()
        else:
            self.attribute = -1
            self.classDistribution = class_probs.copy()
            self.proportion = None
            self.treeNodes = None

            if self.debug:
                print("No useful MSU split found -> leaf created")


@dataclass
class TreeClassifierMSU(TreeClassifier):
    """
    Classifier wrapper extending ``random_tree.TreeClassifier`` with MSU.

    This estimator reuses the corrected base-tree probability fallback for
    unseen categories and empty successor nodes. Its TreeMSU implementation
    stores class frequencies at nonempty internal nodes as well.
    """

    X_msu_discrete_: Optional[np.ndarray] = None
    msu_encoding_: Optional[list[dict[Any, int]]] = None
    use_joblib_discretization: bool = False
    discretization_n_jobs: int = -1

    @staticmethod
    def _is_missing_value(v: Any) -> bool:
        """Return whether a value is missing (``None`` or floating-point NaN).

        Parameters
        ----------
        v : object
            Input value to inspect.

        Returns
        -------
        bool
            ``True`` for a missing value; otherwise ``False``.
        """
        if v is None:
            return True
        try:
            return bool(np.isnan(v))
        except TypeError:
            return False

    def _discretize_with_upo_disc(
            self,
            X: np.ndarray,
            y_int: np.ndarray,
            feature_types: list[str],
            use_joblib: Optional[bool] = None,
            n_jobs: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Construye X_msu_discrete usando upo_disc.FayyadIranisDiscretization
        para atributos numéricos/reales y codificación entera para categóricos.
        
        Devuelve
        --------
        Xd : np.ndarray, shape (n_samples, n_features), dtype=np.int64
        meta : dict
        """
        X = np.asarray(X, dtype=object)
        y_int = np.asarray(y_int, dtype=np.int64)

        n_samples, n_features = X.shape
        Xd = np.zeros((n_samples, n_features), dtype=np.int64)

        categorical_maps = {}
        numeric_indices = []
        
        for j in range(n_features):
            if feature_types[j] == "categorical":
                col = X[:, j]
                values = []
                for v in col:
                    if self._is_missing_value(v):
                        continue
                    if v not in values:
                        values.append(v)

                mapping = {v: i for i, v in enumerate(values)}
                missing_code = len(mapping)

                out = np.empty(n_samples, dtype=np.int64)
                for i, v in enumerate(col):
                    if self._is_missing_value(v):
                        out[i] = missing_code
                    else:
                        out[i] = mapping[v]

                Xd[:, j] = out
                categorical_maps[j] = mapping

            elif feature_types[j] in ("numeric", "real"):
                numeric_indices.append(j)

            else:
                raise ValueError(f"Tipo de atributo desconocido en columna {j}: {feature_types[j]}")

        # OLD CODE:
        # numeric_model = None
        # if len(numeric_indices) > 0:
        #     X_num = np.asarray(X[:, numeric_indices], dtype=float)
        #     disc = ud.FayyadIranisDiscretization(X_num, y_int)
        #     numeric_model = disc.model
        #     X_num_discrete = np.array(
        #         [numeric_model.discretize(row) for row in X_num],
        #         dtype=np.int64
        #     )
        #
        # NEW CODE:
        # The model now transforms the numeric matrix directly, avoiding the
        # row loop after cut points have been learned.
        #
        # JOBLIB OPTION:
        # Fayyad-Irani feature discretization is independent by column. The
        # ud.FayyadIranisDiscretization call below can use joblib when
        # use_joblib=True. It is disabled by default for the lymphoma benchmark
        # because 39-row column tasks are too small for thread scheduling to pay
        # off, but the option is exposed for larger data.
        numeric_model = None
        if len(numeric_indices) > 0:
            X_num = np.asarray(X[:, numeric_indices], dtype=float)

            disc = ud.FayyadIranisDiscretization(
                X_num,
                y_int,
                use_joblib=(
                    self.use_joblib_discretization
                    if use_joblib is None
                    else use_joblib
                ),
                n_jobs=(
                    self.discretization_n_jobs
                    if n_jobs is None
                    else n_jobs
                ),
            )
            numeric_model = disc.model
            # TRAINING PATH TRANSFORM:
            # A forest/tree fit usually transforms each learned discretization
            # once. For that one-shot case, NumPy searchsorted is faster than
            # paying Numba's first-call compilation cost. Users can still call
            # model.transform(..., use_numba=True) for repeated transforms.
            X_num_discrete = numeric_model.transform(X_num, use_numba=False)

            for pos, j in enumerate(numeric_indices):
                Xd[:, j] = X_num_discrete[:, pos]

        meta = {
            "categorical_maps": categorical_maps,
            "numeric_indices": numeric_indices,
            "numeric_model": numeric_model,
        }
            
        return Xd, meta

    def prepare_msu_discrete(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types=None,
        use_joblib: Optional[bool] = None,
        n_jobs: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Precompute the discrete matrix used by MSU.

        Use this when fitting many trees or rerunning benchmarks with the same
        data. Passing the returned Xd to fit(..., X_msu_discrete=Xd,
        rediscretize_msu_per_node=False) skips the discretization cost.
        """
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D.")

        if feature_types is None:
            feature_types = self.infer_feature_types(X)
        else:
            feature_types = list(feature_types)

        _, y_int = np.unique(y, return_inverse=True)
        return self._discretize_with_upo_disc(
            X,
            y_int,
            feature_types,
            use_joblib=use_joblib,
            n_jobs=n_jobs,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types=None,
        feature_names=None,
        sample_weight=None,
        X_msu_discrete: Optional[np.ndarray] = None,
        rediscretize_msu_per_node: bool = True,    
        use_joblib_discretization: Optional[bool] = None,
        discretization_n_jobs: Optional[int] = None,
    ):
        """Fit the MSU-selected decision tree.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Mixed categorical and numeric training features.
        y : array_like of shape (n_samples,)
            Class labels.
        feature_types : list of str or None, optional
            ``"categorical"``, ``"numeric"`` or ``"real"`` for each feature.
        feature_names : list of str or None, optional
            Names used in tree reports.
        sample_weight : array_like of float or None, optional
            Weights applied during split computations (not MSU scoring).
        X_msu_discrete : ndarray of int or None, optional
            Precomputed MSU matrix; used only when node rediscretization is
            disabled. Shape must match ``X``.
        rediscretize_msu_per_node : bool, default True
            If true, fit new Fayyad-Irani bins separately at each tree node.
            If false, reuse the supplied or automatically prepared global bins.
        use_joblib_discretization : bool or None, optional
            Optional override of feature-level parallel discretization.
        discretization_n_jobs : int or None, optional
            Number of discretization workers when joblib is enabled.

        Returns
        -------
        TreeClassifierMSU
            Fitted estimator with a ``TreeMSU`` root stored in ``tree``.
        """
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)

        
        debug_state = {"global_it": 0}
        
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D.")

        if y.ndim != 1:
            raise ValueError("y debe ser un vector 1D.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de instancias.")

        ninstances, nfeatures = X.shape

        row_ids = np.arange(ninstances, dtype=np.int64)
        
        if sample_weight is None:
            weights = np.ones(ninstances, dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.ndim != 1:
                raise ValueError("sample_weight debe ser un vector 1D.")
            if weights.shape[0] != ninstances:
                raise ValueError("sample_weight debe tener la misma longitud que X e y.")

        if feature_types is None:
            feature_types = self.infer_feature_types(X)
        else:
            feature_types = list(feature_types)

        if len(feature_types) != nfeatures:
            raise ValueError(
                f"feature_types debe tener longitud {nfeatures}, "
                f"pero tiene longitud {len(feature_types)}."
            )

        self.feature_types = feature_types
        self.categorical_values = self.get_categorical_values(X, feature_types)

        if feature_names is None:
            self.feature_names = [f"att{i}" for i in range(nfeatures)]
        else:
            if len(feature_names) != nfeatures:
                raise ValueError(
                    f"feature_names debe tener longitud {nfeatures}, "
                    f"pero tiene longitud {len(feature_names)}."
                )
            self.feature_names = list(feature_names)

        self.classes_, y_int = np.unique(y, return_inverse=True)
        nclasses = len(self.classes_)

        # OLD CODE:
        # if rediscretize_msu_per_node:
        #     Xd = None
        #     mappings = None
        # else:
        #     if X_msu_discrete is None:
        #         Xd, mappings = self._discretize_with_upo_disc(
        #             X,
        #             y_int,
        #             self.feature_types,
        #         )
        #     else:
        #         Xd = np.asarray(X_msu_discrete, dtype=np.int64)
        #         mappings = []
        #
        # Later, Xd was recomputed even when rediscretize_msu_per_node=True,
        # so the flag did not do what it said.
        #
        # NEW CODE:
        # Respect the flag. None means rediscretize each node; an ndarray means
        # reuse and propagate one global discrete matrix.
        if rediscretize_msu_per_node:
            Xd = None
            mappings = None
        elif X_msu_discrete is None:
            Xd, mappings = self._discretize_with_upo_disc(
                X,
                y_int,
                self.feature_types,
                use_joblib=use_joblib_discretization,
                n_jobs=discretization_n_jobs,
            )
        else:
            Xd = np.asarray(X_msu_discrete, dtype=np.int64)
            mappings = []


        #print(mappings["numeric_model"])
        #sys.exit(-1)
        
        if Xd is not None and Xd.shape != (ninstances, nfeatures):
            raise ValueError(
                "X_msu_discrete debe tener shape "
                f"({ninstances}, {nfeatures}), pero tiene {Xd.shape}."
            )

        self.X_msu_discrete_ = Xd
        self.msu_encoding_ = mappings
        self.weights = weights

        #if self.debug:
        #    print("=== DEBUG X ===")
        #    print(f"X.shape = {X.shape}")
        #    print("X[:20] =")
        #    print(X[:50])
        #    print("=== DEBUG X_msu_discrete ===")
        #    print(f"Xd.shape = {Xd.shape}")
        #    print("Xd[:20] =")
        # trial_subset = base_subset   print(Xd[:50])
        
        #sys.exit(-1)

        
        if self.computeImpurityDecreasees:
            self.impurityDecreasees = np.zeros((nfeatures, 2), dtype=np.float64)
        else:
            self.impurityDecreasees = None

        if self.kValue > nfeatures:
            self.kValue = nfeatures
        elif self.kValue < 1:
            self.kValue = int(math.log2(nfeatures) + 1) if nfeatures > 1 else 1

        rand = py_random.Random(self.seed)
        
        class_probs = np.bincount(
            y_int,
            weights=weights,
            minlength=nclasses,
        ).astype(np.float64)

        total_weight = float(np.sum(weights))

        if self.debug:
            print("=== DEBUG TreeClassifierMSU.fit() ===")
            print(f"X.shape = {X.shape}")
            if Xd is None:
                print("X_msu_discrete = per-node discretization")
            else:
                print(f"X_msu_discrete.shape = {Xd.shape}")
            print(f"y.shape = {y.shape}")
            print(f"ninstances = {ninstances}")
            print(f"nfeatures = {nfeatures}")
            print(f"kValue = {self.kValue}")
            print(f"classes_ = {self.classes_}")
            print(f"class_probs = {class_probs}")
            print(f"total_weight = {total_weight}")

        tree = TreeMSU(
            debug=self.debug,
            minNum=self.minNum,
            maxDepth=self.maxDepth,
            breakTiesRandomly=self.breakTiesRandomly,
            kValue=self.kValue,
            computeImpurityDecreasees=self.computeImpurityDecreasees,
            random_state=self.random_state,
            seed=self.seed,
            weights=weights,
            feature_types=self.feature_types,
            categorical_values=self.categorical_values,
            n_classes=nclasses,
            msu_subset=None,
        )
        tree.impurityDecreasees = self.impurityDecreasees

        att_indices_window = np.arange(nfeatures, dtype=np.int64)

        tree.build_tree_msu(
            X,
            Xd,
            y_int,
            weights,
            class_probs,
            att_indices_window,
            total_weight,
            rand,
            0,
            None,
            debug_state,
            row_ids,
        )

        self.tree = tree

        if self.debug:
            print("=== END DEBUG TreeClassifierMSU.fit() ===")

        return self
