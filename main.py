import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from similarity.gpt_similarity import calculate_similarity_matrix_gpt
from similarity.tfidf_similarity import calculate_similarity_matrix_tfidf

# Carregar o CSV
df = pd.read_csv('samples/sample_definitions.csv')

# Selecionar uma amostra aleatória
sample_df = df.sample(n=24, random_state=42)

# Obter as definições
definitions = sample_df['Definition'].tolist()

# Salvar a amostra em um novo CSV
sample_df.to_csv('sample_definitions.csv', index=False)

# Calcular a matriz de similaridade usando GPT
similarity_matrix_gpt = calculate_similarity_matrix_gpt(definitions)

# Salvar a matriz de similaridade do GPT em um CSV
pd.DataFrame(similarity_matrix_gpt).to_csv('sample_similarity_matrix_gpt.csv', index=False)

# Calcular a matriz de similaridade usando TF-IDF
similarity_matrix_tfidf = calculate_similarity_matrix_tfidf(definitions)

# Salvar a matriz de similaridade do TF-IDF em um CSV
pd.DataFrame(similarity_matrix_tfidf).to_csv('sample_similarity_matrix_tfidf.csv', index=False)

# Obter os valores mínimo e máximo de ambas as matrizes para normalização
min_value = min(np.min(similarity_matrix_gpt), np.min(similarity_matrix_tfidf))
max_value = max(np.max(similarity_matrix_gpt), np.max(similarity_matrix_tfidf))

# Visualizar a matriz de similaridade (GPT)
cmap = plt.cm.hot_r

plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix_gpt, cmap=cmap, interpolation='nearest', vmin=min_value, vmax=max_value)
plt.xticks(np.arange(0, len(definitions), 1))
plt.yticks(np.arange(0, len(definitions), 1))
plt.colorbar(label='Similarity Score')
plt.title('Heatmap of Similarity Scores (GPT)')
plt.xlabel('Definitions')
plt.ylabel('Definitions')
plt.show()

# Visualizar a matriz de similaridade (TF-IDF)
plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix_tfidf, cmap=cmap, interpolation='nearest', vmin=min_value, vmax=max_value)
plt.xticks(np.arange(0, len(definitions), 1))
plt.yticks(np.arange(0, len(definitions), 1))
plt.colorbar(label='Similarity Score')
plt.title('Heatmap of Similarity Scores (TF-IDF)')
plt.xlabel('Definitions')
plt.ylabel('Definitions')
plt.show()
