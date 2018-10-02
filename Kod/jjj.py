from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

a = pd.DataFrame([[1, 91, 5], [2, 0, 65], [11, 16, 45], [19, 17, 885]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
b = pd.DataFrame([[10, 8, 3], [2, 5, 9], [16, 45, 88], [13, 5, 8]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
c = pd.DataFrame([[4, 6, 11], [4, 82, 0], [31, 1, 2], [15, 3, 3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])

print("a - b pearson by rows:")
print(a.corrwith(b))
print()

cor = np.corrcoef(np.ndarray.flatten(a.as_matrix()), [np.ndarray.flatten(b.as_matrix()), np.ndarray.flatten(c.as_matrix())])
print("a - b - c pearson flatten:")
print(cor)
print()

spearman_table = [[None, None, None] for _ in range(3)]
for i, table_i in enumerate([np.ndarray.flatten(a.as_matrix()), (np.ndarray.flatten(b.as_matrix())), np.ndarray.flatten(c.as_matrix())]):
    for j, table_j in enumerate([np.ndarray.flatten(a.as_matrix()), (np.ndarray.flatten(b.as_matrix())), np.ndarray.flatten(c.as_matrix())]):
        spearman_table[i][j] = spearmanr(table_i, table_j)[0]
print("a - b - c spearman flatten")
print(spearman_table)

def reduce(table):
    reduced_flattened_table = []
    for i in range(len(table)):
        for j in range(i):
            reduced_flattened_table.append(table[i][j])
    print("flattened table:")
    print(reduced_flattened_table)
    return(reduced_flattened_table)

reduced_cor = reduce(cor)
reduced_spe = reduce(spearman_table)

print("pearson - spearman pearson whole flattened table")
print(np.corrcoef(np.ndarray.flatten(cor), np.ndarray.flatten(np.array(spearman_table))))

print("pearson - spearman pearson reduced flattened table")
print(np.corrcoef(reduced_cor, reduced_spe))