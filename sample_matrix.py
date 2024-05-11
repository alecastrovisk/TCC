import pandas as pd
from gpt import get_similarity_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('samples/sample_definitions.csv')

sample_df = df.sample(n=24, random_state=42)

definitions = sample_df['Definition'].tolist()

sample_df.to_csv('sample_definitions.csv', index=False)

n = len(definitions)
similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i+1, n):
        similarity = get_similarity_score(definitions[i], definitions[j])
        similarity_matrix[i, j] = similarity

pd.DataFrame(similarity_matrix).to_csv('sample_similarity_matrix.csv', index=False)

cmap = plt.cm.hot_r


plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap=cmap, interpolation='nearest', vmin=np.min(similarity_matrix), vmax=np.max(similarity_matrix))

plt.xticks(np.arange(0, n, 1))
plt.yticks(np.arange(0, n, 1))
plt.colorbar(label='Similarity Score')
plt.title('Heatmap of Similarity Scores')
plt.xlabel('Definitions')
plt.ylabel('Definitions')
plt.show()
