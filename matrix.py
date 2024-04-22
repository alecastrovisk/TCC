import pandas as pd
from gpt import get_similarity_score
import numpy as np
import matplotlib.pyplot as plt
import os

# Carregar o dataframe
df = pd.read_csv('samples/tests_smells.csv')
max = len(df)
print('Tamanho do dataframe:', max)

# Verificar se o arquivo de progresso e a matriz existem
progress_file = 'progress.csv'
matrix_file = 'matriz_parcial.csv'

if os.path.exists(progress_file):
    progress_df = pd.read_csv(progress_file)
    last_i, last_j = progress_df.values[0]
else:
    last_i, last_j = 0, 0

if os.path.exists(matrix_file):
    mat = np.loadtxt(matrix_file, delimiter=',')
else:
    mat = np.zeros((max, max))

definitions = list(df['Definition'])

for i in range(last_i, max):
    start_j = last_j if i == last_i else i + 1  # Começar de i + 1 para evitar comparações repetidas
    for j in range(start_j, max):
        def1 = definitions[i]
        def2 = definitions[j]
        result = get_similarity_score(def1, def2)
        mat[i][j] = result

        # Salvar o progresso e a matriz parcial a cada 10 iterações
        if j % 10 == 0:
            progress_df = pd.DataFrame([[i, j]], columns=['i', 'j'])
            progress_df.to_csv(progress_file, index=False)
            np.savetxt(matrix_file, mat, delimiter=',', fmt='%.2f')

    last_j = 0  # Resetar last_j após a primeira iteração de i

# Remover o arquivo de progresso após a conclusão
if os.path.exists(progress_file):
    os.remove(progress_file)

df_mat = pd.DataFrame(mat)

df_mat.to_csv('matriz_similaridade.csv', index=False)

cmap = plt.cm.hot_r

plt.figure(figsize=(10, 8))
plt.imshow(df_mat, cmap=cmap, interpolation='nearest', vmin=0, vmax=100)

plt.xticks(np.arange(0, max, 1))
plt.yticks(np.arange(0, max, 1))
plt.colorbar(label='Similarity Score')
plt.title('Heatmap of Similarity Scores')
plt.xlabel('Definitions')
plt.ylabel('Definitions')
plt.show()
