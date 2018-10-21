# import editdistance
from Tools import *
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# from AST import *
# from functools import partial
# from matplotlib import pyplot as plt
# import ast
from random import randint, choice


a = pd.DataFrame([[1, 91, 8], [6, 0, 0], [1, np.nan, 0], [5, 17, 9]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
b = pd.DataFrame([[10, 8, 3], [2, 5, 9], [16, 45, 88], [13, 5, 8]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
c = pd.DataFrame([[4, 6, 11], [4, 82, 0], [31, 1, 2], [15, 3, 3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
d = pd.DataFrame([["flrsRI/Wxkbdyg{}><=!0123456789", 6, 1], ["lIf/1", 82, 1], ["056123!>/", 1, 1], ["Ir///", 3, 2]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])



e = a.one * a.three
print(e)
























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
