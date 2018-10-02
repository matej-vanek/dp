import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
mat = [[0.3871582746059306, 0.22444081299152283, 0.05676982262622136, 0.3316310897763252],
       [0.286546451364201, 0.332997815565169, 0.032552362187861394, 0.3479033708827687],
       [0.31404444444444446, 0.1375111111111111, 0.15777777777777777, 0.39066666666666666],
       [0.2591210449005518, 0.21426019294136328, 0.05674434607699128, 0.4698744160810936]]

for row in mat:
    print(sum(row))

idx = ["quick_few", "quick_many", "slow_few", "slow_many"]
cols = idx
df = pd.DataFrame(mat, index=idx, columns=cols)

# _r reverses the normal order of the color map 'RdYlGn'
sns.heatmap(df, cmap='Blues', annot=True)
plt.xlabel("Target group")
plt.ylabel("Source group")
plt.title("Total relativized transition matrix of Robomission users")
plt.show()
