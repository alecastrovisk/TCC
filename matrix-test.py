import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Exemplo de definições e matriz (substitua pelo seu código)
definitions = ['definição 1', 'definição 2', 'definição 3']
mat = [[0.8, 0.3, 0.6],
       [0.3, 1.0, 0.2],
       [0.6, 0.2, 0.9]]

# Convertendo a matriz para um DataFrame pandas
df = pd.DataFrame(mat, columns=definitions, index=definitions)

# Criando um mapa de calor usando matplotlib
plt.figure(figsize=(10, 8))
plt.title('Matriz de Similaridade')
heatmap = plt.imshow(df, cmap='viridis', interpolation='nearest')
plt.colorbar(heatmap)
plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45)
plt.yticks(np.arange(len(df.index)), df.index)
plt.tight_layout()
plt.show()
