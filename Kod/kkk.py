from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import re
from functools import partial
from Tools import *

a = pd.DataFrame([[1, 91, 5], [0, 0, 65], [1, 16, 45], [0, 17, 5]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
b = pd.DataFrame([[10, 8, 3], [2, 5, 9], [16, 45, 88], [13, 5, 8]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
c = pd.DataFrame([[4, 6, 11], [4, 82, 0], [31, 1, 2], [15, 3, 3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])
d = pd.DataFrame([["flrsRI/Wxkbdyg{}><=!0123456789", 6, 1], ["lIf/1", 82, 1], ["056123!>/", 1, 1], ["Ir///", 3, 2]], index=['a', 'b', 'c', 'd'], columns=['one', 'two', 'three'])

"""
print(a)
print(a.corr())
sns.heatmap(a.corr(), cmap='viridis', annot=True, vmin=-1, vmax=1)
plt.show()

print(np.corrcoef(np.ndarray.flatten(np.matrix([[1.000000, 0.908224, -0.70099], [0.908224, 1.000000, -0.59692], [-0.700990, -0.596920, 1.00000]])),
                  np.ndarray.flatten(np.matrix([[1.000000, 0.805850, -0.692362], [0.805850, 1.000000, -0.610929], [-0.692362, -0.610929, 1.000000]]))))
"""
"""
q = [[ 1.,          0.99922828], [ 0.99922828,  1.        ]]
w = [[ 1.,          0.99899567], [ 0.99899567,  1.        ]]
print(np.corrcoef(np.ndarray.flatten(np.array(q)), np.ndarray.flatten(np.array(w))))
"""
x = np.median(list(map(lambda x: len(re.sub(pattern="[{}0123456789<>=!]", repl="", string=x)), d["one"])))
#x = np.median(list(map(len, list(map(re.sub("[{}0123456789<>=!]", "", d["one"]))))))
print(x)
#print(median_of_lens(d["one"]))