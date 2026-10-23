"""Random Forest inspirado en Weka con árboles ponderados y OOB.

El archivo contiene las clases ``Tree``, ``TreeClassifier`` y
``RandomForest`` (alias ``RandomForestClassifier``). Define sus propios
árboles base: NO importa ni hereda de ``random_tree.py``. La copia
interna del árbol NO incorpora las dos correcciones recientes de
``random_tree_vi.py`` (distribuciones en nodos internos y ramas vacías).

Dependencies
------------
NumPy, stat_utils (con alternativa local para entropía), y joblib opcional.

Notes
-----
Las opciones del bosque no son equivalentes en todos los detalles al
RandomForestClassifier de scikit-learn; tampoco implementan por sí solas
un criterio MSU de selección de atributos.

Authors
-------
- Miguel Garcia Torres <mgarciat@upo.es>
- Thomas Vanhaeren

License
-------
See the project LICENSE file.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
import numpy as np
import sys
import math
import random

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    Parallel = None
    delayed = None
    JOBLIB_AVAILABLE = False

try:
    import stat_utils as ul
except ImportError:
    class _ULFallback:
        @staticmethod
        def entropy_fromhistogram(hist):
            import numpy as _np
            hist = _np.asarray(hist, dtype=float)
            total = float(_np.sum(hist))
            if total <= 0.0:
                return 0.0
            p = hist[hist > 0.0] / total
            return float(-_np.sum(p * _np.log2(p)))

        @staticmethod
        def conditional_entropy_onrows(dist):
            import numpy as _np
            dist = _np.asarray(dist, dtype=float)
            total = float(_np.sum(dist))
            if total <= 0.0:
                return 0.0
            value = 0.0
            for row in dist:
                row_total = float(_np.sum(row))
                if row_total > 0.0:
                    value += (row_total / total) * _ULFallback.entropy_fromhistogram(row)
            return float(value)

    ul = _ULFallback()

@dataclass
class Tree:
    # Configuration
    """Nodo de árbol de decisión con divisiones numéricas/categóricas, pesos y valores ausentes.

    Notes
    -----
    Configuración de un nodo y almacenamiento de su estado aprendido.
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
        """
        Equivalente aproximado a weka.core.Utils.gr(a, b):
        true si a es claramente mayor que b.
        """
        return (a - b) > Tree.WEKA_SMALL

    @staticmethod
    def weka_eq(a, b):
        """
        Equivalente aproximado a weka.core.Utils.eq(a, b).
        """
        return abs(a - b) <= Tree.WEKA_SMALL

    @staticmethod
    def is_missing_value(v):
        """Implementación auxiliar de is missing value en Tree.

        Parameters
        ----------
        v : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        if v is None:
            return True
        try:
            return bool(np.isnan(v))
        except TypeError:
            return False

    def gain(self, dist, prior_val):
        """
        Equivalent to:
            priorVal - ContingencyTables.entropyConditionedOnRows(dist)
        """
        return prior_val - ul.conditional_entropy_onrows(dist)

    def prior_val(self, dist):
        """
        Equivalent to:
            ContingencyTables.entropyOverColumns(dist)
        """
        dist = np.asarray(dist, dtype=float)
        class_counts = np.sum(dist, axis=0)
        return ul.entropy_fromhistogram(class_counts)

    def distribution(self, props, dists, att, X, y, weights):
        """
        Classification distribution for a candidate split attribute.
        Updates:
            props[0]
            dists[0]
        Returns:
            split_point
        """

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
        """
        Simplified port of Weka splitData(data), without sample_weight.
        Returns:
            [(X0, y0, w0), (X1, y1, w1), ...]
        """

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
        """
        Normaliza una distribución de clases.
        
        En las hojas, classDistribution contiene pesos/conteos por clase.
        distributionForInstance debe devolver probabilidades.
        """
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
        """
        Replica la lógica de Weka JRandomTree.distributionForInstance(instance).
        
        Devuelve
        --------
        np.ndarray, shape (n_classes,)
        Probabilidad estimada de pertenencia a cada clase.
        
        Reglas:
        - Si el nodo es hoja, devuelve classDistribution normalizada.
        - Si el atributo de split está perdido, combina las distribuciones de
        todos los hijos ponderando por self.proportion.
        - Si el atributo es categórico, baja por la rama del valor nominal.
        - Si el atributo es numérico, baja por:
        rama 0 si value < splitPoint
        rama 1 si value >= splitPoint
        """
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

        if child is None:
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

        """Construir recursivamente el árbol, atendiendo a los pesos, la pureza y la profundidad.

        Parameters
        ----------
        X : object
            Consulte las anotaciones y el comportamiento de la implementación.
        y : object
            Consulte las anotaciones y el comportamiento de la implementación.
        weights : object
            Consulte las anotaciones y el comportamiento de la implementación.
        class_probs : object
            Consulte las anotaciones y el comportamiento de la implementación.
        att_indices_window : object
            Consulte las anotaciones y el comportamiento de la implementación.
        total_weight : object
            Consulte las anotaciones y el comportamiento de la implementación.
        random : object
            Consulte las anotaciones y el comportamiento de la implementación.
        depth : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
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
        """
        Devuelve la representación de una hoja en formato parecido a Weka:
            : clase (total/errores)
        """

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
        """
        Formatea números como Weka, evitando decimales innecesarios.
        """
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
        """
        Devuelve el árbol en formato textual parecido a Weka.
        """

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
        """
        Número total de nodos del árbol.
        """
        if self.attribute == -1 or self.treeNodes is None:
            return 1

        return 1 + sum(child.num_nodes() for child in self.treeNodes if child is not None)


    def num_leaves(self):
        """
        Número de hojas del árbol.
        """
        if self.attribute == -1 or self.treeNodes is None:
            return 1

        return sum(child.num_leaves() for child in self.treeNodes if child is not None)

    def used_attributes(self):
        """
        Devuelve un conjunto con los índices de atributos usados en nodos internos.
        """
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
        """
        Devuelve árbol + resumen:
            Size of the tree
            Number of leaves
            Number of attributes used
            Attributes used
        """

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
        """Implementación auxiliar de single variance en TreeClassifier.

        Parameters
        ----------
        total_sum : object
            Consulte las anotaciones y el comportamiento de la implementación.
        total_sum_squared : object
            Consulte las anotaciones y el comportamiento de la implementación.
        total_weight : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        if total_weight <= 0:
            return 0.0
        return total_sum_squared - (total_sum ** 2) / total_weight

    @staticmethod
    def is_missing_value(v):
        """Implementación auxiliar de is missing value en TreeClassifier.

        Parameters
        ----------
        v : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        if v is None:
            return True
        try:
            return bool(np.isnan(v))
        except TypeError:
            return False

    def infer_feature_types(self, X):
        """Implementación auxiliar de infer feature types en TreeClassifier.

        Parameters
        ----------
        X : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
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
        """Implementación auxiliar de get categorical values en TreeClassifier.

        Parameters
        ----------
        X : object
            Consulte las anotaciones y el comportamiento de la implementación.
        feature_types : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
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
        """Ajustar los modelos a X e y. Las clases de entrenamiento se almacenan en classes_.

        Parameters
        ----------
        X : object
            Consulte las anotaciones y el comportamiento de la implementación.
        y : object
            Consulte las anotaciones y el comportamiento de la implementación.
        feature_types : object
            Consulte las anotaciones y el comportamiento de la implementación.
        feature_names : object
            Consulte las anotaciones y el comportamiento de la implementación.
        sample_weight : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
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
        """
        Devuelve el árbol en formato parecido a Weka.
        """

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
        """Implementación auxiliar de num nodes en TreeClassifier.

        Parameters
        ----------

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        if self.tree is None:
            raise ValueError("El árbol no está entrenado. Llama primero a fit().")
        return self.tree.num_nodes()

    def num_leaves(self):
        """Implementación auxiliar de num leaves en TreeClassifier.

        Parameters
        ----------

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        if self.tree is None:
            raise ValueError("El árbol no está entrenado. Llama primero a fit().")
        return self.tree.num_leaves()

    def used_attributes(self):
        """Implementación auxiliar de used attributes en TreeClassifier.

        Parameters
        ----------

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        if self.tree is None:
            raise ValueError("El árbol no está entrenado. Llama primero a fit().")
        return self.tree.used_attributes()

    def used_attribute_names(self):
        """Implementación auxiliar de used attribute names en TreeClassifier.

        Parameters
        ----------

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        used = sorted(self.used_attributes())

        if self.feature_names is None:
            return [f"att{i}" for i in used]

        return [self.feature_names[i] for i in used]


    def distribution_for_instance(self, x):
        """
        Devuelve la distribución de probabilidad para una única instancia.
        
        Equivalente a distributionForInstance(instance) en Weka.
        """
        if self.tree is None:
            raise ValueError("El árbol no está entrenado. Llama primero a fit().")

        x = np.asarray(x, dtype=object)

        if x.ndim != 1:
            raise ValueError("x debe ser una única instancia 1D.")

        return self.tree.distribution_for_instance(x)


    def predict_proba(self, X):
        """
        Devuelve probabilidades por clase para una matriz de instancias.
        
        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
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
        """
        Predice la clase mayoritaria usando predict_proba.
        """
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]


@dataclass
class RandomForest:
    """
    Port parcial de ``weka.classifiers.trees.RandomForest`` usando el
    ``TreeClassifier`` definido en este módulo como árbol base.

    Equivalencias principales con Weka
    ----------------------------------
    - ``numIterations`` equivale a ``-I``: número de árboles. Por defecto, 100.
    - ``kValue`` equivale a ``numFeatures`` / ``KValue`` del RandomTree.
      Si ``kValue < 1``, el árbol base lo ajusta a ``int(log2(n_features) + 1)``.
    - ``bagSizePercent`` equivale a ``-P``: tamaño de cada bolsa como porcentaje
      del conjunto de entrenamiento. Por defecto, 100.
    - Las copias bootstrap se representan mediante pesos, como hace Weka con
      ``setRepresentCopiesUsingWeights(true)``.
    - La predicción por defecto promedia las distribuciones de probabilidad de
      los árboles, que es el comportamiento habitual de Bagging/RandomForest.

    Notas
    -----
    Esta clase está pensada para clasificación. Mantiene una interfaz cercana a
    scikit-learn: ``fit``, ``predict_proba``, ``predict``, ``get_params`` y
    ``set_params``.
    """

    # Forest configuration
    numIterations: int = 100
    bagSizePercent: int = 100
    seed: int = 1
    random_state: Optional[int] = None
    majority_vote: bool = False
    calcOutOfBag: bool = False
    n_jobs: int = 1
    parallel_backend: str = "loky"

    # Base RandomTree configuration
    debug: bool = False
    minNum: float = 1.0
    maxDepth: int = 0
    breakTiesRandomly: bool = False
    kValue: int = 0
    computeAttributeImportance: bool = False

    # Learned state
    estimators_: List[TreeClassifier] = field(default_factory=list, init=False)
    bootstrap_indices_: List[np.ndarray] = field(default_factory=list, init=False)
    bootstrap_counts_: List[np.ndarray] = field(default_factory=list, init=False)
    classes_: Optional[np.ndarray] = field(default=None, init=False)
    feature_types: Optional[list] = field(default=None, init=False)
    feature_names: Optional[list] = field(default=None, init=False)
    n_features_in_: Optional[int] = field(default=None, init=False)
    oob_decision_function_: Optional[np.ndarray] = field(default=None, init=False)
    oob_score_: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        """Validar los parámetros de configuración al crear el bosque.

        Parameters
        ----------

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        if self.numIterations <= 0:
            raise ValueError("numIterations debe ser mayor que 0.")
        if self.bagSizePercent <= 0:
            raise ValueError("bagSizePercent debe ser mayor que 0.")
        if self.parallel_backend not in {"loky", "threading"}:
            raise ValueError("parallel_backend debe ser 'loky' o 'threading'.")

    def _rng_seed(self) -> int:
        """Obtener semilla efectiva respetando random_state si se especifica.

        Parameters
        ----------

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        return self.seed if self.random_state is None else self.random_state

    @staticmethod
    def _validate_X_y(X, y=None):
        """Implementación auxiliar de validate X y en RandomForest.

        Parameters
        ----------
        X : object
            Consulte las anotaciones y el comportamiento de la implementación.
        y : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        X = np.asarray(X, dtype=object)
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D.")
        if y is None:
            return X
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y debe ser un vector 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de instancias.")
        return X, y

    def _make_tree(self, tree_seed: int) -> TreeClassifier:
        """Crear un árbol base con los parámetros y semilla suministrados.

        Parameters
        ----------
        tree_seed : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        return TreeClassifier(
            debug=self.debug,
            minNum=self.minNum,
            maxDepth=self.maxDepth,
            breakTiesRandomly=self.breakTiesRandomly,
            kValue=self.kValue,
            computeImpurityDecreasees=self.computeAttributeImportance,
            random_state=tree_seed,
            seed=tree_seed,
        )

    def _fit_tree_from_bootstrap_spec(
        self,
        spec,
        X,
        y,
        base_weight,
        feature_types,
        feature_names,
    ):
        """Ajustar un árbol según índices y conteos de una muestra bootstrap.

        Parameters
        ----------
        spec : object
            Consulte las anotaciones y el comportamiento de la implementación.
        X : object
            Consulte las anotaciones y el comportamiento de la implementación.
        y : object
            Consulte las anotaciones y el comportamiento de la implementación.
        base_weight : object
            Consulte las anotaciones y el comportamiento de la implementación.
        feature_types : object
            Consulte las anotaciones y el comportamiento de la implementación.
        feature_names : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        i, indices, counts, tree_seed = spec
        weights = base_weight * counts
        tree = self._make_tree(tree_seed)
        tree.fit(
            X,
            y,
            feature_types=feature_types,
            feature_names=feature_names,
            sample_weight=weights,
        )
        return i, tree, indices, counts, tree_seed

    def _aligned_tree_proba(self, tree: TreeClassifier, X: np.ndarray) -> np.ndarray:
        """
        Devuelve probabilidades alineadas con ``self.classes_``.

        Un bootstrap puede no contener alguna clase minoritaria. En ese caso, el
        árbol base tendrá menos columnas de probabilidad; aquí se recolocan en
        las columnas globales del bosque.
        """
        local_proba = tree.predict_proba(X)
        aligned = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        class_to_global = {c: i for i, c in enumerate(self.classes_)}
        for local_idx, cls in enumerate(tree.classes_):
            global_idx = class_to_global[cls]
            aligned[:, global_idx] = local_proba[:, local_idx]
        return aligned

    def fit(self, X, y, feature_types=None, feature_names=None, sample_weight=None):
        """Ajustar los modelos a X e y. Las clases de entrenamiento se almacenan en classes_.

        Parameters
        ----------
        X : object
            Consulte las anotaciones y el comportamiento de la implementación.
        y : object
            Consulte las anotaciones y el comportamiento de la implementación.
        feature_types : object
            Consulte las anotaciones y el comportamiento de la implementación.
        feature_names : object
            Consulte las anotaciones y el comportamiento de la implementación.
        sample_weight : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        X, y = self._validate_X_y(X, y)
        n_instances, n_features = X.shape
        self.n_features_in_ = n_features

        if sample_weight is None:
            base_weight = np.ones(n_instances, dtype=np.float64)
        else:
            base_weight = np.asarray(sample_weight, dtype=np.float64)
            if base_weight.shape[0] != n_instances:
                raise ValueError("sample_weight debe tener la misma longitud que X e y.")

        self.classes_ = np.unique(y)
        self.feature_types = list(feature_types) if feature_types is not None else None
        self.feature_names = list(feature_names) if feature_names is not None else [f"att{i}" for i in range(n_features)]

        bag_size = int(round(n_instances * (self.bagSizePercent / 100.0)))
        bag_size = max(1, bag_size)

        rng = np.random.default_rng(self._rng_seed())
        self.estimators_ = []
        self.bootstrap_indices_ = []
        self.bootstrap_counts_ = []

        if self.debug:
            print("=== DEBUG RandomForest.fit() ===")
            print(f"X.shape = {X.shape}")
            print(f"numIterations = {self.numIterations}")
            print(f"bagSizePercent = {self.bagSizePercent}")
            print(f"bag_size = {bag_size}")
            print(f"seed = {self._rng_seed()}")

        if self.n_jobs != 1 and JOBLIB_AVAILABLE and self.numIterations > 1:
            # JOBLIB FOREST PARALLELISM:
            # Old behavior trained every bootstrap tree sequentially below.
            # New opt-in behavior pre-generates the same deterministic
            # bootstrap indices and tree seeds, fits independent trees in
            # parallel, then sorts by tree index so estimator order matches the
            # sequential path.
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
                delayed(self._fit_tree_from_bootstrap_spec)(
                    spec,
                    X,
                    y,
                    base_weight,
                    feature_types,
                    feature_names,
                )
                for spec in specs
            )

            for i, tree, indices, counts, tree_seed in sorted(results, key=lambda item: item[0]):
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
            weights = base_weight * counts

            # Cada árbol recibe una semilla independiente pero reproducible.
            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            tree = self._make_tree(tree_seed)
            tree.fit(
                X,
                y,
                feature_types=feature_types,
                feature_names=feature_names,
                sample_weight=weights,
            )

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

    def _compute_oob_score(self, X, y):
        """Estimar el acierto out-of-bag usando únicamente árboles donde cada observación quedó fuera del bootstrap.

        Parameters
        ----------
        X : object
            Consulte las anotaciones y el comportamiento de la implementación.
        y : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        n_instances = X.shape[0]
        n_classes = len(self.classes_)
        sums = np.zeros((n_instances, n_classes), dtype=np.float64)
        counts = np.zeros(n_instances, dtype=np.float64)

        for tree, boot_counts in zip(self.estimators_, self.bootstrap_counts_):
            oob_mask = boot_counts == 0
            if not np.any(oob_mask):
                continue
            proba = self._aligned_tree_proba(tree, X[oob_mask])
            sums[oob_mask] += proba
            counts[oob_mask] += 1.0

        valid = counts > 0
        oob = np.full((n_instances, n_classes), np.nan, dtype=np.float64)
        oob[valid] = sums[valid] / counts[valid, None]
        self.oob_decision_function_ = oob

        if np.any(valid):
            pred = self.classes_[np.nanargmax(oob[valid], axis=1)]
            self.oob_score_ = float(np.mean(pred == y[valid]))
        else:
            self.oob_score_ = np.nan

    def predict_proba(self, X):
        """Calcular probabilidades estimadas por clase para observaciones nuevas.

        Parameters
        ----------
        X : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        if not self.estimators_:
            raise ValueError("El bosque no está entrenado. Llama primero a fit().")
        X = self._validate_X_y(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X tiene {X.shape[1]} atributos, pero el modelo fue entrenado con "
                f"{self.n_features_in_}."
            )

        if self.majority_vote:
            votes = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
            class_to_global = {c: i for i, c in enumerate(self.classes_)}
            for tree in self.estimators_:
                pred = tree.predict(X)
                for row, cls in enumerate(pred):
                    votes[row, class_to_global[cls]] += 1.0
            return votes / float(len(self.estimators_))

        proba_sum = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for tree in self.estimators_:
            proba_sum += self._aligned_tree_proba(tree, X)
        proba = proba_sum / float(len(self.estimators_))

        row_sums = proba.sum(axis=1)
        nonzero = row_sums > 0.0
        proba[nonzero] = proba[nonzero] / row_sums[nonzero, None]
        return proba

    def predict(self, X):
        """Predecir las clases de nuevas observaciones.

        Parameters
        ----------
        X : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def distribution_for_instance(self, x):
        """Calcular la distribución por clase de una observación.

        Parameters
        ----------
        x : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        x = np.asarray(x, dtype=object)
        if x.ndim != 1:
            raise ValueError("x debe ser una única instancia 1D.")
        return self.predict_proba(x.reshape(1, -1))[0]

    def get_estimators(self):
        """Implementación auxiliar de get estimators en RandomForest.

        Parameters
        ----------

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        return self.estimators_

    def getClassifiers(self):
        """Alias de estilo Weka."""
        return self.get_estimators()

    def compute_average_impurity_decrease_per_attribute(self):
        """Promediar la reducción de impureza por atributo en los árboles.

        Parameters
        ----------

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        if not self.computeAttributeImportance:
            raise ValueError("computeAttributeImportance=False; no se han recogido estadísticas.")
        if not self.estimators_:
            raise ValueError("El bosque no está entrenado.")

        n_features = self.n_features_in_
        total_decrease = np.zeros(n_features, dtype=np.float64)
        total_count = np.zeros(n_features, dtype=np.float64)

        for tree in self.estimators_:
            imp = tree.impurityDecreasees
            if imp is None:
                continue
            total_decrease += imp[:, 0]
            total_count += imp[:, 1]

        avg = np.zeros(n_features, dtype=np.float64)
        mask = total_count > 0.0
        avg[mask] = total_decrease[mask] / total_count[mask]
        return avg, total_count

    def computeAverageImpurityDecreasePerAttribute(self, node_counts=None):
        """Alias cercano al nombre de Weka."""
        avg, counts = self.compute_average_impurity_decrease_per_attribute()
        if node_counts is not None:
            node_counts[:] = counts
        return avg

    def compute_average_metrics(self):
        """
        Devuelve medias y desviaciones típicas de complejidad del bosque.

        Incluye las seis medidas del método Java ``computeAverageMetrics``:
        atributos usados, nodos internos/ramas y hojas, cada una con media y std.
        Además añade el tamaño total del árbol para comodidad.
        """
        if not self.estimators_:
            return None

        attrs = np.asarray([len(t.used_attributes()) for t in self.estimators_], dtype=float)
        leaves = np.asarray([t.num_leaves() for t in self.estimators_], dtype=float)
        nodes = np.asarray([t.num_nodes() for t in self.estimators_], dtype=float)
        internal = nodes - leaves

        def mean_std(v):
            return float(np.mean(v)), float(np.std(v, ddof=0))

        attr_mean, attr_std = mean_std(attrs)
        internal_mean, internal_std = mean_std(internal)
        leaves_mean, leaves_std = mean_std(leaves)
        nodes_mean, nodes_std = mean_std(nodes)

        return {
            "attributes_used_mean": attr_mean,
            "attributes_used_std": attr_std,
            "internal_nodes_mean": internal_mean,
            "internal_nodes_std": internal_std,
            "leaves_mean": leaves_mean,
            "leaves_std": leaves_std,
            "tree_size_mean": nodes_mean,
            "tree_size_std": nodes_std,
        }

    def computeAverageMetrics(self):
        """
        Alias de estilo Weka.

        Devuelve un array compatible con el método Java:
        [mean_attrs, std_attrs, mean_internal_nodes, std_internal_nodes,
         mean_leaves, std_leaves]
        """
        m = self.compute_average_metrics()
        if m is None:
            return None
        return np.asarray([
            m["attributes_used_mean"],
            m["attributes_used_std"],
            m["internal_nodes_mean"],
            m["internal_nodes_std"],
            m["leaves_mean"],
            m["leaves_std"],
        ], dtype=np.float64)

    def get_params(self, deep=True):
        """Recuperar parámetros configurables como diccionario.

        Parameters
        ----------
        deep : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        return {
            "numIterations": self.numIterations,
            "bagSizePercent": self.bagSizePercent,
            "seed": self.seed,
            "random_state": self.random_state,
            "majority_vote": self.majority_vote,
            "calcOutOfBag": self.calcOutOfBag,
            "n_jobs": self.n_jobs,
            "parallel_backend": self.parallel_backend,
            "debug": self.debug,
            "minNum": self.minNum,
            "maxDepth": self.maxDepth,
            "breakTiesRandomly": self.breakTiesRandomly,
            "kValue": self.kValue,
            "computeAttributeImportance": self.computeAttributeImportance,
        }

    def set_params(self, **params):
        """Modificar parámetros de configuración, rechazando los desconocidos.

        Parameters
        ----------
        **params : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        valid = self.get_params().keys()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Parámetro desconocido: {key}")
            setattr(self, key, value)
        return self

    def to_string(self, print_classifiers=False, num_decimal_places=2):
        """Implementación auxiliar de to string en RandomForest.

        Parameters
        ----------
        print_classifiers : object
            Consulte las anotaciones y el comportamiento de la implementación.
        num_decimal_places : object
            Consulte las anotaciones y el comportamiento de la implementación.

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        if not self.estimators_:
            return "RandomForest: No model built yet."

        lines = ["RandomForest", "", f"Number of trees: {len(self.estimators_)}"]
        lines.append(f"Bag size percent: {self.bagSizePercent}")
        lines.append(f"KValue/numFeatures: {self.kValue}")
        lines.append(f"n_jobs: {self.n_jobs}")
        lines.append(f"parallel_backend: {self.parallel_backend}")

        if self.calcOutOfBag:
            lines.append(f"Out-of-bag score: {self.oob_score_}")

        metrics = self.compute_average_metrics()
        if metrics is not None:
            lines.append("")
            lines.append("Average forest metrics")
            for key, value in metrics.items():
                lines.append(f"{key}: {value:.{num_decimal_places}f}")

        if self.computeAttributeImportance:
            avg, counts = self.compute_average_impurity_decrease_per_attribute()
            lines.append("")
            lines.append("Attribute importance based on average impurity decrease")
            order = np.argsort(avg)[::-1]
            for idx in order:
                name = self.feature_names[idx] if self.feature_names is not None else f"att{idx}"
                lines.append(f"{avg[idx]:.{num_decimal_places}f} ({counts[idx]:.0f}) {name}")

        if print_classifiers:
            for i, tree in enumerate(self.estimators_, start=1):
                lines.append("")
                lines.append(f"Tree {i}")
                lines.append(tree.to_string(num_decimal_places=num_decimal_places))

        return "\n".join(lines)

    def __str__(self):
        """Implementación auxiliar de str en RandomForest.

        Parameters
        ----------

        Notes
        -----
        Conserva la semántica de la implementación original.
        """
        return self.to_string(print_classifiers=False)


# Alias más explícito para uso estilo scikit-learn.
RandomForestClassifier = RandomForest
