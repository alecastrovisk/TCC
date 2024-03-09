import pandas as pd
from gpt import get_similarity_score
import numpy as np
import random
import matplotlib.pyplot as plt

df = pd.read_csv('samples/tests_smells.csv')

max = len(df)
max = 4

definitions = list(df['Definition'])

rows = max
cols = max
 
mat = [[0 for _ in range(rows)] for _ in range(cols)]

for i in range(0, max):
    for j in range(i+1, max):
        def1 = definitions[i]
        def2 = definitions[j]
        # result = get_similarity_score(def1, def2)
        result = (round(random.random(), 2), 'yes')
        mat[i][j] = result[0]
        print(mat)

df_mat = pd.DataFrame(mat)

cmap = plt.cm.hot_r

plt.figure(figsize=(10, 8))
plt.imshow(df_mat, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
plt.colorbar(label='Similarity Score')
plt.title('Heatmap of Similarity Scores')
plt.xlabel('Definitions')
plt.ylabel('Definitions')
plt.show()