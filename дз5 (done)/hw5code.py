"""
Под критерием Джини здесь подразумевается следующая функция:
$$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
$R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
$H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

Указания:
* Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
* В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
* Поведение функции в случае константного признака может быть любым.
* При одинаковых приростах Джини нужно выбирать минимальный сплит.
* За наличие в функции циклов балл будет снижен. Векторизуйте! :)

:param feature_vector: вещественнозначный вектор значений признака
:param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

:return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
    разделить на две различные подвыборки, или поддерева
:return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
:return threshold_best: оптимальный порог (число)
:return gini_best: оптимальное значение критерия Джини (число)
"""

import numpy as np
from collections import Counter


def H(target_vector, indices):
    p_1 = np.sum(target_vector[indices]) / len(indices)
    p_0 = 1 - p_1
    return 1 - p_0 ** 2 - p_1 ** 2


def find_best_split(feature_vector, target_vector):

    uniques = np.unique(feature_vector)
    thresholds = (uniques[:-1] + uniques[1:]) / 2

    ginis = []
    for threshold in thresholds:
        g_i = np.argwhere(feature_vector > threshold)
        l_i = np.argwhere(feature_vector < threshold)
        ginis.append(- len(l_i) / len(feature_vector) * H(target_vector, l_i) - len(g_i) / len(feature_vector) * H(target_vector, g_i))

    best_index = np.argmax(ginis)
    
    return (thresholds, ginis, thresholds[best_index], ginis[best_index])


class DecisionTree:
    def __init__(self, feature_types, max_depth=np.inf, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if depth == self._max_depth or len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split, map_ = None, None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]

            elif feature_type == "categorical":
                categories_map = {}
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))

            f_uniques = np.unique(feature_vector)
            if len(f_uniques) > 1:
                _, _, threshold, gini = find_best_split(feature_vector, sub_y)
                if gini_best is None or gini > gini_best:
                    feature_best = feature
                    gini_best = gini
                    threshold_best = threshold
                    split = feature_vector < threshold
                    map_ = categories_map

        if feature_best is None or split.sum() < self._min_samples_leaf or len(sub_y) - split.sum() < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._feature_types[feature_best] == 'categorical':
            node['map'] = map_ 

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
       
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)


    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        if self._feature_types[node['feature_split']] == 'categorical':
            aux = node['map'][x[node['feature_split']]]
        else:
            aux = x[node['feature_split']]
        
        if aux < node["threshold"]:
            return self._predict_node(x, node['left_child'])

        else:
            return self._predict_node(x, node['right_child'])

    
    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)


    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
