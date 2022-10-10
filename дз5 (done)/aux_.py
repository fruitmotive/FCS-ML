from collections import Counter
import numpy as np

x = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
y = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0])

categories_map = {}
counts = Counter(x)
clicks = Counter(x[y == 1])
ratio = {}
for key, current_count in counts.items():
    if key in clicks:
        current_click = clicks[key]
    else:
        current_click = 0
    ratio[key] = current_click / current_count
sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
feature_vector = np.array(list(map(lambda x: categories_map[x], x)))

print(feature_vector)