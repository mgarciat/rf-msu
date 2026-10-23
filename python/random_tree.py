"""Weka-inspired weighted random-tree classification for mixed feature types.

This module provides :class:`Tree` and :class:`TreeClassifier`.
Information gain is calculated using :mod:`stat_utils` (bits).

Authors
-------
- Miguel Garcia Torres <mgarciat@upo.es>
- Thomas Vanhaeren

Notes
-----
The implementation is distinct from scikit-learn's CART trees; identical
splits are not guaranteed on general datasets.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
import numpy as np
import sys
import math
import random
import stat_utils as ul

@dataclass
class Tree:
    # Configuration
    """Recursive random decision-tree node storing weighted splits and predictions.

Attributes
----------
debug : bool
    Enable diagnostic printing.
minNum : float
    Minimum leaf weight criterion.
maxDepth : int
    Maximum depth; zero means no fixed limit.
kValue : int
    Number of candidate attributes (normalized during fitting).
seed : int
    Random seed used in fitting.
"""
    debug: bool = True
    minNum: float = 1.0
    maxDepth: int = 0
    breakTiesRandomly: bool = False
    kValue: int = 0
    computeImpurityDecreasees: bool = True
    random_state: Optional[int] = None
    seed: int = 3417
    weights: Optional[np.ndarray] = None

    #
    last_sorted_indices: Optional[np.ndarray] = field(default=None, init=False)
    last_sorted_att: int = field(default=-1, init=False)
    
    # Metadata
    feature_types: Optional[list] = None
    categorical_values: Optional[dict] = None
    n_classes: Optional[int] = None
    
    
    # Learned tree state
    treeNodes: Optional[List["Tree"]] = field(default=None, init=False)
    attribute: int = field(default=-1, init=False)
    splitPoint: float = field(default=np.nan, init=False)
    proportion: Optional[np.ndarray] = field(default=None, init=False)
    classDistribution: Optional[np.ndarray] = field(default=None, init=False)

    # Indexed by attribute: [sum impurity decrease, count]
    impurityDecreasees: Optional[np.ndarray] = field(default=None, init=False)
    

    WEKA_SMALL = 1e-6

    @staticmethod
    def weka_gr(a, b):
        """Return whether ``a`` exceeds ``b`` by Weka tolerance.

Parameters
----------
a : float
    First value.
b : float
    Second value.

Returns
-------
bool
    Whether ``a - b > WEKA_SMALL``."""
        return (a - b) > Tree.WEKA_SMALL

    @staticmethod
    def weka_eq(a, b):
        """Compare two floating-point numbers with Weka tolerance.

Parameters
----------
a : float
    First value.
b : float
    Second value.

Returns
-------
bool
    Whether the absolute difference is at most ``WEKA_SMALL``."""
        return abs(a - b) <= Tree.WEKA_SMALL

    @staticmethod
    def is_missing_value(v):
        """Detect missing values, including ``None`` and IEEE NaN.

Parameters
----------
v : object
    Candidate value.

Returns
-------
bool
    Whether the value is considered missing."""
        if v is None:
            return True
        try:
            return bool(np.isnan(v))
        except TypeError:
            return False

    def gain(self, dist, prior_val):
        """Compute information gain for a candidate split.

Parameters
----------
dist : array_like
    Branch-by-class weighted contingency table.
prior_val : float
    Parent entropy in bits.

Returns
-------
float
    Parent entropy minus branch-conditional entropy in bits."""
        return prior_val - ul.conditional_entropy_onrows(dist)

    def prior_val(self, dist):
        """Compute class entropy from a branch-by-class table.

Parameters
----------
dist : array_like
    Weighted contingency table with classes in columns.

Returns
-------
float
    Entropy of marginal class distribution in bits."""
        dist = np.asarray(dist, dtype=float)
        class_counts = np.sum(dist, axis=0)
        return ul.entropy_fromhistogram(class_counts)

    def distribution(self, props, dists, att, X, y, weights):
        """Find a candidate split and corresponding weighted class distributions.

Parameters
----------
props : list
    Mutable one-item list receiving branch probabilities.
dists : list
    Mutable one-item list receiving branch-by-class counts.
att : int
    Index of candidate feature.
X : array_like
    Input observations.
y : array_like
    Integer-coded class labels.
weights : array_like
    Nonnegative observation weights.

Returns
-------
float
    Numeric split threshold, or NaN for categorical features.

Notes
-----
Missing values may be treated with weighted branch proportions.
The behavior follows this implementation and may differ from scikit-learn."""

        if self.feature_types is None:
            raise ValueError("self.feature_types no puede ser None.")

        if self.n_classes is None:
            raise ValueError("self.n_classes no puede ser None. Debe fijarse desde TreeClassifier.")

        split_point = np.nan

        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)

        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights debe tener la misma longitud que X e y.")

        n_instances = X.shape[0]
        n_classes = self.n_classes

        index_of_first_missing_value = n_instances
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
                    if index_of_first_missing_value == n_instances:
                        index_of_first_missing_value = i
                    continue

                if att_val not in value_to_index:
                    raise ValueError(
                        f"Valor categórico desconocido en atributo {att}: {att_val}"
                    )

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

            sorted_non_missing_indices = non_missing_indices[
                np.argsort(col[non_missing_indices])
            ]

            sorted_indices = np.concatenate(
                [sorted_non_missing_indices, missing_indices]
            )

            self.last_sorted_indices = sorted_indices.copy()
            self.last_sorted_att = att

            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]
            weights_sorted = weights[sorted_indices]
            col_sorted = col[sorted_indices]

            index_of_first_missing_value = len(sorted_non_missing_indices)

            # Move all non-missing instances into the second subset
            for j in range(index_of_first_missing_value):
                cls = int(y_sorted[j])
                curr_dist[1, cls] += weights_sorted[j]

            prior_val = self.prior_val(curr_dist)

            # Save initial distribution
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

        # Compute branch proportions
        props[0] = np.sum(dist, axis=1).astype(float)

        total_props = float(np.sum(props[0]))

        if np.isclose(total_props, 0.0):
            props[0][:] = 1.0 / len(props[0])
        else:
            props[0] = props[0] / total_props

        # Distribute missing values
        for i in range(n_instances):
            att_val = X[i, att]

            if self.is_missing_value(att_val):
                cls = int(y[i])
                w = weights[i]

                for j in range(dist.shape[0]):
                    dist[j, cls] += w * props[0][j]

        dists[0] = dist
        return split_point

    def split_data(self, X, y, weights):
        """Partition observations into children using the fitted split.

Parameters
----------
X : array_like
    Input observations.
y : array_like
    Class labels.
weights : array_like
    Observation weights.

Returns
-------
list of tuple
    Per-branch ``(X, y, weights)`` arrays; missing samples are distributed fractionally.

Notes
-----
Missing values may be treated with weighted branch proportions.
The behavior follows this implementation and may differ from scikit-learn."""

        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        weights = np.asarray(weights, dtype=np.float64)

        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights debe tener la misma longitud que X e y.")

        if self.attribute < 0:
            raise ValueError("No se puede dividir: self.attribute == -1")

        if self.proportion is None:
            raise ValueError("No se puede dividir: self.proportion is None")

        n_branches = len(self.proportion)

        X_subsets = [[] for _ in range(n_branches)]
        y_subsets = [[] for _ in range(n_branches)]
        w_subsets = [[] for _ in range(n_branches)]
        
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
                        y_subsets[k].append(y[i])
                        w_subsets[k].append(weights[i] * self.proportion[k])
                continue

            if attribute_type == "categorical":
                if value not in value_to_index:
                    raise ValueError(
                        f"Valor categórico desconocido en atributo {self.attribute}: {value}"
                    )

                branch = value_to_index[value]
                X_subsets[branch].append(X[i].copy())
                y_subsets[branch].append(y[i])
                w_subsets[branch].append(weights[i])
                continue

            if attribute_type in ("numeric", "real"):
                branch = 0 if float(value) < self.splitPoint else 1
                X_subsets[branch].append(X[i])
                y_subsets[branch].append(y[i])
                w_subsets[branch].append(weights[i])
                continue

            raise ValueError(f"Tipo de atributo desconocido: {attribute_type}")

        subsets = []

        for k in range(n_branches):
            if len(X_subsets[k]) == 0:
                Xk = np.empty((0, X.shape[1]), dtype=object)
                yk = np.empty((0,), dtype=y.dtype)
                wk = np.empty((0,), dtype=np.float64)
            else:
                Xk = np.asarray(X_subsets[k], dtype=object)
                yk = np.asarray(y_subsets[k], dtype=y.dtype)
                wk = np.asarray(w_subsets[k], dtype=np.float64)
                
            subsets.append((Xk, yk, wk))

        return subsets

    def _normalized_distribution(self, dist):
        """Normalize weighted class frequencies into class probabilities.

Parameters
----------
dist : array_like or None
    Class-frequency vector or missing distribution.

Returns
-------
numpy.ndarray
    Probability vector, or zeros/uniform fallback where specified."""
        if dist is None:
            if self.n_classes is None:
                raise ValueError("No se puede crear distribución: n_classes es None.")
            return np.zeros(self.n_classes, dtype=np.float64)
        
        dist = np.asarray(dist, dtype=np.float64).copy()

        if dist.ndim != 1:
            raise ValueError("La distribución de clases debe ser un vector 1D.")

        total = float(np.sum(dist))

        if total <= 0.0:
            # Caso raro: hoja vacía o rama sin instancias.
            # Weka suele proteger estos casos usando la distribución del nodo padre.
            # Si no hay información, devolvemos distribución uniforme.
            if self.n_classes is None:
                raise ValueError("No se puede normalizar distribución vacía sin n_classes.")
            return np.ones(self.n_classes, dtype=np.float64) / self.n_classes

        return dist / total


    def distribution_for_instance(self, x):
        """Predict class probabilities for a single observation.

Parameters
----------
x : array_like
    One row of features, including possible missing values.

Returns
-------
numpy.ndarray
    Estimated probabilities ordered by trained classes.

Notes
-----
Missing values may be treated with weighted branch proportions.
The behavior follows this implementation and may differ from scikit-learn."""
        x = np.asarray(x, dtype=object)
        
        if x.ndim != 1:
            raise ValueError("x debe ser una instancia 1D.")

        # Caso hoja
        if self.attribute == -1 or self.treeNodes is None:
            return self._normalized_distribution(self.classDistribution)

        att = self.attribute
        value = x[att]
        attribute_type = self.feature_types[att]

        # Caso valor perdido: mezcla de hijos ponderada por self.proportion
        if self.is_missing_value(value):
            if self.proportion is None:
                return self._normalized_distribution(self.classDistribution)

            result = np.zeros(self.n_classes, dtype=np.float64)

            for i, prop in enumerate(self.proportion):
                if prop <= 0.0:
                    continue

                child = self.treeNodes[i]

                if child is None:
                    # Rama vacía: usar distribución almacenada en el nodo actual
                    child_dist = self._normalized_distribution(self.classDistribution)
                else:
                    child_dist = child.distribution_for_instance(x)

                result += prop * child_dist

            total = float(np.sum(result))

            if total <= 0.0:
                return self._normalized_distribution(self.classDistribution)

            return result / total

        # Caso atributo categórico
        if attribute_type == "categorical":
            values = self.categorical_values[att]
            value_to_index = {v: i for i, v in enumerate(values)}
            
            if value not in value_to_index:
                # Valor no visto en entrenamiento.
                # Mejor fallback: distribución del nodo actual.
                return self._normalized_distribution(self.classDistribution)
            
            branch = value_to_index[value]

            # Caso atributo numérico
        elif attribute_type in ("numeric", "real"):
            branch = 0 if float(value) < self.splitPoint else 1

        else:
            raise ValueError(f"Tipo de atributo desconocido: {attribute_type}")

        # Bajar por la rama correspondiente
        if branch < 0 or branch >= len(self.treeNodes):
            return self._normalized_distribution(self.classDistribution)

        child = self.treeNodes[branch]

        # Si la rama no existe o no contiene observaciones de entrenamiento,
        # emplear la distribución de clases del nodo padre. Esto evita
        # devolver un vector de probabilidades nulo para una rama vacía.
        if child is None or child.classDistribution is None:
            return self._normalized_distribution(self.classDistribution)

        return child.distribution_for_instance(x)

    
    def build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        class_probs: np.ndarray,
        att_indices_window: np.ndarray,
        total_weight: float,
        random: Any,
        depth: int
    ) -> None:

        """Recursively grow an information-gain random tree.

Parameters
----------
X : array_like
    Node feature matrix.
y : array_like
    Integer class labels.
weights : array_like
    Sample weights.
class_probs : array_like
    Weighted class counts for this node.
att_indices_window : numpy.ndarray
    Mutable randomized attribute window.
total_weight : float
    Nominal node weight; recomputed internally.
random : random.Random
    Pseudorandom generator.
depth : int
    Current depth.

Returns
-------
None
    Tree state is updated in place."""
        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)
        class_probs = np.asarray(class_probs, dtype=float)

        self.weights = weights
        if weights.shape[0] != X.shape[0]:
            raise ValueError("weights debe tener la misma longitud que X e y.")
        
        ninstances, nfeatures = X.shape

        if ninstances == 0:
            self.attribute = -1
            self.classDistribution = None
            self.proportion = None
            return

        # Conservar los conteos ponderados de clases también en nodos internos.
        # Permite un fallback probabilístico válido para categorías no vistas.
        # Se mantiene la convención previa para nodos vacíos (None).
        self.classDistribution = class_probs.copy()
        
        total_weight = float(np.sum(class_probs))

        if self.debug:
            if not np.isclose(total_weight, np.sum(weights)):
                if self.debug:
                    print(
                        f"Warning: total_weight={total_weight}, "
                        f"sum(weights)={np.sum(weights)}"
                    )
        
        is_pure_classification_node = np.isclose(
            class_probs[np.argmax(class_probs)],
            total_weight
        )

        max_depth_reached = (
            self.maxDepth > 0 and depth >= self.maxDepth
        )

        if self.debug:
            print("==========build tree==========")
            print(f"total_weight = {total_weight}")
            print(f"minNum = {self.minNum}")
            print(f"is_pure_classification_node = {is_pure_classification_node}")
            print(f"max_depth_reached = {max_depth_reached}")
            print(f"depth = {depth}, maxDepth = {self.maxDepth}")

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

            if self.debug:
                print(f"classDistribution = {self.classDistribution}")

            return

        val = -np.inf
        split = np.nan
        best_dists = None
        best_props = None
        best_index = 0

        props = [None]
        dists = [None]

        window_size = nfeatures
        k = self.kValue
        gain_found = False

        #att_indices_window = np.arange(nfeatures)
        if self.debug:
            print(f"att_indices_window after swap = {att_indices_window}")

        best_sorted_indices = None
        while window_size > 0 and ((k > 0) or (not gain_found)):

            k -= 1
            if self.debug:
                print("\t\n--- while attributes ---")
                print(f"\t window_size = {window_size}")
                print(f"\t k before decrement = {k}")
                print(f"\t gain_found = {gain_found}")

            

            chosen_index = random.randrange(window_size)
            att_index = int(att_indices_window[chosen_index])

            if self.debug:
                print(f"\t chosen_index = {chosen_index}")
                print(f"\t att_index = {att_index}")

            att_indices_window[chosen_index] = att_indices_window[window_size - 1]
            att_indices_window[window_size - 1] = att_index
            window_size -= 1

            if self.debug:
                print(f"\t att_indices_window after swap = {att_indices_window}")
                print(f"\t window_size after decrement = {window_size}")

            curr_split = self.distribution(props, dists, att_index, X, y, weights)
            curr_val = self.gain(dists[0], self.prior_val(dists[0]))

            if self.debug:
                print(f"\t curr_split = {curr_split}")
                print(f"\t curr_val = {curr_val}")

            if self.weka_gr(curr_val, 0.0):
                gain_found = True
                if self.debug:
                    print("\t gain_found set to True")

            if (
                (curr_val > val)
                or (
                    (not self.breakTiesRandomly)
                    and (curr_val == val)
                    and (att_index < best_index)
                )
            ):
                val = curr_val
                best_index = att_index
                split = curr_split
                best_props = props[0].copy()
                best_dists = dists[0].copy()

                if self.last_sorted_indices is not None and self.last_sorted_att == att_index:
                    best_sorted_indices = self.last_sorted_indices.copy()
                else:
                    best_sorted_indices = None
                
                if self.debug:
                    print("\t Best attribute updated")
                    print(f"\t val = {val}")
                    print(f"\t best_index = {best_index}")
                    print(f"\t split = {split}")
                    print(f"\t best_props = {best_props}")
                    print(f"\t best_dists = {best_dists}")

        self.attribute = best_index

        if self.debug:
            print(f"best_index = {best_index}")
            print(f"val = {val}")

        if self.weka_gr(val, 0.0):
            if self.computeImpurityDecreasees and self.impurityDecreasees is not None:
                self.impurityDecreasees[self.attribute][0] += val
                self.impurityDecreasees[self.attribute][1] += 1

                if self.debug:
                    print(
                        f"Updated impurityDecreasees[{self.attribute}] = "
                        f"{self.impurityDecreasees[self.attribute]}"
                    )

            self.splitPoint = split
            self.proportion = best_props

            #
            if best_sorted_indices is not None:
            #if (best_sorted_indices is not None
            #    and self.last_sorted_att == self.attribute
            #):
                X_for_split = X[best_sorted_indices]
                y_for_split = y[best_sorted_indices]
                weights_for_split = weights[best_sorted_indices]
            else:
                X_for_split = X
                y_for_split = y
                weights_for_split = weights
            
            
            subsets = self.split_data(X_for_split, y_for_split, weights_for_split)

            self.treeNodes = [None] * len(best_dists)

            if self.debug:
                print(f"self.splitPoint = {self.splitPoint}")
                print(f"self.proportion = {self.proportion}")
                print(f"len(subsets) = {len(subsets)}")
                print(f"len(best_dists) = {len(best_dists)}")
                
            for i in range(len(best_dists)):
                X_child, y_child, w_child = subsets[i]

                child = Tree(
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
                )

                child.impurityDecreasees = self.impurityDecreasees


                child_total_weight = float(np.sum(w_child))
#                child_total_weight = float(np.sum(best_dists[i]))

                if self.debug:
                    print(f"Building child {i}, child_total_weight = {child_total_weight}")

                child.build_tree(
                    X_child, #subsets[i][0],
                    y_child, #subsets[i][1],
                    w_child,
                    best_dists[i],
                    att_indices_window, #                    att_indices_window.copy(),
                    child_total_weight,
                    random,
                    depth + 1
                )

                self.treeNodes[i] = child

            empty_successor = False

            for i in range(len(subsets)):
                X_child, y_child, w_child = subsets[i]

                if X_child.shape[0] == 0:
                    empty_successor = True
                    break

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
                print("No useful split found -> leaf created")
                
    def _leaf_string(self, class_names=None, num_decimal_places=2):
        """Format a leaf with class, total weight, and errors.

Parameters
----------
class_names : sequence or None
    Optional labels for class indices.
num_decimal_places : int
    Number of decimal places.

Returns
-------
str
    Weka-style leaf label."""

        if self.classDistribution is None:
            return " : None"

        dist = np.asarray(self.classDistribution, dtype=float)

        total = float(np.sum(dist))

        if total <= 0:
            return " : None (0/0)"

        max_index = int(np.argmax(dist))
        max_count = float(dist[max_index])
        errors = total - max_count

        if class_names is not None:
            class_label = str(class_names[max_index])
        else:
            class_label = str(max_index)

        return (
            f" : {class_label} "
            f"({total:.{num_decimal_places}f}/{errors:.{num_decimal_places}f})"
        )

    def _format_number(self, value, num_decimal_places=2):
        """Format numeric thresholds in Weka-style output.

Parameters
----------
value : float or None
    Value to format.
num_decimal_places : int
    Precision of output.

Returns
-------
str
    Human-readable formatted number."""
        if value is None:
            return "None"

        if isinstance(value, float) and np.isnan(value):
            return "NaN"

        text = f"{float(value):.{num_decimal_places}f}"

        # Quitar ceros finales: 2.50 -> 2.5, 2.00 -> 2
        text = text.rstrip("0").rstrip(".")

        return text


    def to_string(
        self,
        feature_names=None,
        class_names=None,
        level=0,
        num_decimal_places=2
    ):
        """Render the fitted tree as readable Weka-style text.

Parameters
----------
feature_names : sequence or None
    Optional feature names.
class_names : sequence or None
    Optional class names (Tree only).
level : int
    Initial indentation depth (Tree only).
num_decimal_places : int
    Decimal precision.

Returns
-------
str
    Tree representation or tree summary."""

        # Hoja
        if self.attribute == -1 or self.treeNodes is None:
            return self._leaf_string(
                class_names=class_names,
                num_decimal_places=num_decimal_places
            )

        if feature_names is None:
            att_name = f"att{self.attribute}"
        else:
            att_name = str(feature_names[self.attribute])

        attribute_type = self.feature_types[self.attribute]

        text = ""

        if attribute_type in ("numeric", "real"):
            split_text = self._format_number(
                self.splitPoint,
                num_decimal_places=num_decimal_places
            )

            # Rama izquierda: < split
            text += "\n"
            text += "|   " * level
            text += f"{att_name} < {split_text}"
            text += self.treeNodes[0].to_string(
                feature_names=feature_names,
                class_names=class_names,
                level=level + 1,
                num_decimal_places=num_decimal_places
            )

            # Rama derecha: >= split
            text += "\n"
            text += "|   " * level
            text += f"{att_name} >= {split_text}"
            text += self.treeNodes[1].to_string(
                feature_names=feature_names,
                class_names=class_names,
                level=level + 1,
                num_decimal_places=num_decimal_places
            )

        elif attribute_type == "categorical":
            values = self.categorical_values[self.attribute]

            for i, value in enumerate(values):
                text += "\n"
                text += "|   " * level
                text += f"{att_name} = {value}"
                text += self.treeNodes[i].to_string(
                    feature_names=feature_names,
                    class_names=class_names,
                    level=level + 1,
                    num_decimal_places=num_decimal_places
                )

        else:
            raise ValueError(f"Tipo de atributo desconocido: {attribute_type}")

        return text


    def num_nodes(self):
        """Count all nodes in the fitted tree.

Returns
-------
int
    Total number of nodes."""
        if self.attribute == -1 or self.treeNodes is None:
            return 1

        return 1 + sum(child.num_nodes() for child in self.treeNodes if child is not None)


    def num_leaves(self):
        """Count all leaves in the fitted tree.

Returns
-------
int
    Total number of terminal nodes."""
        if self.attribute == -1 or self.treeNodes is None:
            return 1

        return sum(child.num_leaves() for child in self.treeNodes if child is not None)

    def used_attributes(self):
        """Collect indices of features used for internal splits.

Returns
-------
set of int
    Set of used feature indices."""
        if self.attribute == -1 or self.treeNodes is None:
            return set()

        attrs = {self.attribute}

        for child in self.treeNodes:
            if child is not None:
                attrs.update(child.used_attributes())

        return attrs


    def summary_string(
        self,
        feature_names=None,
        class_names=None,
        num_decimal_places=2
    ):
        """Return the tree representation and structural summary.

Parameters
----------
feature_names : sequence or None
    Names of input features.
class_names : sequence or None
    Names of classes.
num_decimal_places : int
    Numeric precision.

Returns
-------
str
    Tree text followed by counts and attribute names."""

        tree_text = self.to_string(
            feature_names=feature_names,
            class_names=class_names,
            level=0,
            num_decimal_places=num_decimal_places
        )

        # Quitar salto inicial, porque to_string empieza con "\n" en nodos internos
        tree_text = tree_text.lstrip("\n")

        used = sorted(self.used_attributes())

        if feature_names is None:
            used_names = [f"att{i}" for i in used]
        else:
            used_names = [str(feature_names[i]) for i in used]

        summary = "\n\n"
        summary += f"Size of the tree : {self.num_nodes()}\n"
        summary += f"Number of leaves : {self.num_leaves()}\n"
        summary += f"Number of attributes used : {len(used)}\n"

        if len(used_names) > 0:
            summary += "Attributes used : " + ", ".join(used_names)
        else:
            summary += "Attributes used : None"

        return tree_text + summary

@dataclass
class TreeClassifier:
    """
    Classifier wrapper for the Tree class.
    """

    debug: bool = True
    minNum: float = 1.0
    maxDepth: int = 0
    breakTiesRandomly: bool = False
    kValue: int = 0
    computeImpurityDecreasees: bool = True
    random_state: Optional[int] = None
    seed: int = 3417

    impurityDecreasees: Optional[np.ndarray] = field(default=None, init=False)
    tree: Optional["Tree"] = field(default=None, init=False)
    classes_: Optional[np.ndarray] = field(default=None, init=False)
    feature_types: Optional[list] = field(default=None, init=False)
    categorical_values: Optional[dict] = field(default=None, init=False)
    feature_names: Optional[list] = field(default=None, init=False)
    weights: Optional[np.ndarray] = None

    
    @staticmethod
    def single_variance(total_sum, total_sum_squared, total_weight):
        """Compute a weighted unnormalized variance numerator.

Parameters
----------
total_sum : float
    Weighted sum of values.
total_sum_squared : float
    Weighted sum of squared values.
total_weight : float
    Sum of weights.

Returns
-------
float
    ``sum(w*x**2) - sum(w*x)**2/sum(w)``; zero when weight is nonpositive."""
        if total_weight <= 0:
            return 0.0
        return total_sum_squared - (total_sum ** 2) / total_weight

    @staticmethod
    def is_missing_value(v):
        """Detect missing values, including ``None`` and IEEE NaN.

Parameters
----------
v : object
    Candidate value.

Returns
-------
bool
    Whether the value is considered missing."""
        if v is None:
            return True
        try:
            return bool(np.isnan(v))
        except TypeError:
            return False

    def infer_feature_types(self, X):
        """Infer numeric or categorical types from columns.

Parameters
----------
X : array_like
    Two-dimensional input data.

Returns
-------
list of str
    Per-feature type labels."""
        feature_types = []

        for j in range(X.shape[1]):
            col = X[:, j]

            is_numeric = True
            for v in col:
                if self.is_missing_value(v):
                    continue

                try:
                    float(v)
                except (ValueError, TypeError):
                    is_numeric = False
                    break

            feature_types.append("numeric" if is_numeric else "categorical")

        return feature_types

    def get_categorical_values(self, X, feature_types):
        """List observed categories for every categorical column.

Parameters
----------
X : array_like
    Input feature matrix.
feature_types : sequence of str
    Feature type labels.

Returns
-------
dict
    Map from feature index to sorted observed category values."""
        categorical_values = {}

        for j, ftype in enumerate(feature_types):
            if ftype == "categorical":
                vals = []

                for v in X[:, j]:
                    if self.is_missing_value(v):
                        continue
                    if v not in vals:
                        vals.append(v)

                try:
                    vals = sorted(vals, key=lambda z: float(z))
                except Exception:
                    vals = sorted(vals, key=str)

                categorical_values[j] = vals

        return categorical_values
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types=None,
        feature_names=None,
        sample_weight=None    
    ):
        """Train the random-tree classifier.

Parameters
----------
X : array_like
    Two-dimensional training predictors.
y : array_like
    One-dimensional target labels.
feature_types : sequence or None
    Optional per-column types.
feature_names : sequence or None
    Optional feature names.
sample_weight : array_like or None
    Optional observation weights.

Returns
-------
TreeClassifier
    Fitted estimator (self).

Notes
-----
Missing values may be treated with weighted branch proportions.
The behavior follows this implementation and may differ from scikit-learn."""
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D.")

        if y.ndim != 1:
            raise ValueError("y debe ser un vector 1D.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de instancias.")

        ninstances, nfeatures = X.shape
        
        if sample_weight is None:
            weights = np.ones(ninstances, dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape[0] != ninstances:
                raise ValueError("sample_weight debe tener la misma longitud que X e y.")

        
        if self.debug:
            print("=== DEBUG fit() ===")
            print(f"X.shape = {X.shape}")
            print(f"y.shape = {y.shape}")
            print(f"ninstances = {ninstances}")
            print(f"nfeatures = {nfeatures}")
            print(f"initial kValue = {self.kValue}")
            print(f"computeImpurityDecreasees = {self.computeImpurityDecreasees}")
            print(f"seed = {self.seed}")

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
        
        if self.computeImpurityDecreasees:
            self.impurityDecreasees = np.zeros((nfeatures, 2), dtype=np.float64)
        else:
            self.impurityDecreasees = None

        if self.debug:
            print(f"feature_types = {self.feature_types}")
            print(f"categorical_values = {self.categorical_values}")
            if self.impurityDecreasees is not None:
                print(f"impurityDecreasees.shape = {self.impurityDecreasees.shape}")

        if self.kValue > nfeatures:
            self.kValue = nfeatures
            if self.debug:
                print(f"kValue too large -> adjusted to {self.kValue}")
        elif self.kValue < 1:
            self.kValue = int(math.log2(nfeatures) + 1) if nfeatures > 1 else 1
            if self.debug:
                print(f"kValue < 1 -> adjusted to {self.kValue}")

        rand = random.Random(self.seed)

        self.classes_, y_int = np.unique(y, return_inverse=True)
        nclasses = len(self.classes_)

        if self.debug:
            print("Entering classification branch")
            print(f"classes_ = {self.classes_}")
            print(f"y_int[:20] = {y_int[:20]}")


        class_probs = np.bincount(
            y_int,
            weights=weights,
            minlength=nclasses
        ).astype(np.float64)
        
        totalWeight = float(np.sum(weights))

        if self.debug:
            print(f"class_probs = {class_probs}", flush=True)
            print(f"totalWeight = {totalWeight}", flush=True)
        
        
        
        tree = Tree(
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
            n_classes=nclasses
        )
        att_indices_window = np.arange(nfeatures)
        tree.impurityDecreasees = self.impurityDecreasees

        tree.build_tree(
            X,
            y_int,
            weights,
            class_probs,
            att_indices_window, #            att_indices_window.copy(),
            totalWeight,
            rand,
            0
        )

        self.tree = tree

        if self.debug:
            print("=== END DEBUG fit() ===")

        return self
    
    
    def to_string(self, feature_names=None, num_decimal_places=2):
        """Render the fitted tree as readable Weka-style text.

Parameters
----------
feature_names : sequence or None
    Optional feature names.
num_decimal_places : int
    Decimal precision.

Returns
-------
str
    Tree representation or tree summary."""

        if self.tree is None:
            raise ValueError("El árbol no está entrenado. Llama primero a fit().")

        if feature_names is None:
            feature_names = self.feature_names

        return self.tree.summary_string(
            feature_names=feature_names,
            class_names=self.classes_,
            num_decimal_places=num_decimal_places
        )

    def num_nodes(self):
        """Count all nodes in the fitted tree.

Returns
-------
int
    Total number of nodes."""
        if self.tree is None:
            raise ValueError("El árbol no está entrenado. Llama primero a fit().")
        return self.tree.num_nodes()

    def num_leaves(self):
        """Count all leaves in the fitted tree.

Returns
-------
int
    Total number of terminal nodes."""
        if self.tree is None:
            raise ValueError("El árbol no está entrenado. Llama primero a fit().")
        return self.tree.num_leaves()

    def used_attributes(self):
        """Collect indices of features used for internal splits.

Returns
-------
set of int
    Set of used feature indices."""
        if self.tree is None:
            raise ValueError("El árbol no está entrenado. Llama primero a fit().")
        return self.tree.used_attributes()

    def used_attribute_names(self):
        """Return names of features used for splits.

Returns
-------
list of str
    Names of used attributes."""
        used = sorted(self.used_attributes())

        if self.feature_names is None:
            return [f"att{i}" for i in used]

        return [self.feature_names[i] for i in used]


    def distribution_for_instance(self, x):
        """Predict class probabilities for a single observation.

Parameters
----------
x : array_like
    One row of features, including possible missing values.

Returns
-------
numpy.ndarray
    Estimated probabilities ordered by trained classes.

Notes
-----
Missing values may be treated with weighted branch proportions.
The behavior follows this implementation and may differ from scikit-learn."""
        if self.tree is None:
            raise ValueError("El árbol no está entrenado. Llama primero a fit().")

        x = np.asarray(x, dtype=object)

        if x.ndim != 1:
            raise ValueError("x debe ser una única instancia 1D.")

        return self.tree.distribution_for_instance(x)


    def predict_proba(self, X):
        """Predict class probabilities for one or more observations.

Parameters
----------
X : array_like
    Input feature matrix or single row.

Returns
-------
numpy.ndarray
    Array of shape ``(n_samples, n_classes)``.

Notes
-----
Missing values may be treated with weighted branch proportions.
The behavior follows this implementation and may differ from scikit-learn."""
        if self.tree is None:
            raise ValueError("El árbol no está entrenado. Llama primero a fit().")

        X = np.asarray(X, dtype=object)

        if X.ndim == 1:
            return self.distribution_for_instance(X).reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D o una instancia 1D.")

        return np.vstack([
            self.distribution_for_instance(X[i])
            for i in range(X.shape[0])
        ])


    def predict(self, X):
        """Predict class labels using maximum estimated probability.

Parameters
----------
X : array_like
    Input observations.

Returns
-------
numpy.ndarray
    Predicted labels in the original target-label space."""
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]
