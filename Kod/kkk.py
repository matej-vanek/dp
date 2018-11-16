# import editdistance
# from Tools import *
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# from AST import *
# from functools import partial
from matplotlib import pyplot as plt
# import ast
from random import randint, choice
import pandas as pd
import numpy as np

a = pd.DataFrame([[1, 91, 8], [6, 0, 0], [1, np.nan, 8], [5, 17, 9]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
b = pd.DataFrame([[10, 8, 3], [2, 5, 9], [16, 45, 88], [13, 5, 8]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
c = pd.DataFrame([[4, 6, 11], [4, 82, 0], [31, 1, 2], [15, 3, 3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
d = pd.DataFrame([["flrsRI/Wxkbdyg{}><=!0123456789", 6, 1], ["lIf/1", 82, 1], ["056123!>/", 1, 1], ["Ir///", 3, 2]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])

a.one = pd.Series([1,2,3,4], index=a.index)

print(a)
"""
from mpl_toolkits.mplot3d import Axes3D

abs_step = 1
abs_begin = 1
abs_end = 7
rel_step = 10
rel_begin = 1
rel_end = 5
abs_thresholds = [abs_step * i for i in range(abs_begin, abs_end)]
rel_thresholds = [rel_step * i for i in range(rel_begin, rel_end)]
frequents = [[] for _ in range(len(rel_thresholds))]
for i, rel_threshold in enumerate(rel_thresholds):
    print(i)
    for abs_threshold in abs_thresholds:
        frequents[i].append(abs_threshold*rel_threshold)

abs_axis = np.array([abs_thresholds for _ in range(len(rel_thresholds))])
rel_axis = np.array([[item for _ in range(len(abs_thresholds))] for item in rel_thresholds])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(abs_axis, rel_axis, np.array(frequents), color="b")
ax.set_xlabel("absolute count threshold")
ax.set_ylabel("relative count threshold")
ax.set_zlabel("frequent wrong programs ratio")
ax.set_title("kkk")
plt.show()
"""























"""
q = np.array(["", "f", "ssl", "ssss", "R4{ssss}", "R4{f}", "R5{f}", "W!b{R5{f}}", "W!b{lR3{r}}", "W!b{rR3{rf}}"])
w = np.array(list(map(partial(build_ast, verbose=False), q)))
print(w)
#print(ast_ted(build_ast(q[6]), build_ast(q[7])))

dist_matrix = []
for i in range(len(w)):
    for j in range(len(w)):
        if i < j:
            dist_matrix.append(ast_ted(w[i], w[j]))
print(dist_matrix)
flat = np.ndarray.flatten(np.array(dist_matrix))

Z = linkage(flat)
print(Z)

y = fcluster(Z, 5, criterion="distance")
print(y)
print("Number of found clusters: ", len(set(y)))

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90,  # rotates the x axis labels
    leaf_font_size=10,  # font size for the x axis labels
)
plt.show()
"""
