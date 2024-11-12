import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from similarity.gpt_similarity import calculate_similarity_matrix_gpt
from similarity.tfidf_similarity import calculate_similarity_matrix_tfidf

def load_csv(file_path):
    """
    Carrega um arquivo CSV.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Erro ao carregar o CSV: {e}")
        return None

def save_csv(df, file_path):
    """
    Salva um DataFrame em um arquivo CSV.
    """
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Erro ao salvar o CSV: {e}")

def plot_heatmap(matrix, title, definitions, min_value, max_value, cmap=plt.cm.hot_r):
    """
    Plota um heatmap para uma matriz de similaridade.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=min_value, vmax=max_value)
    plt.xticks(np.arange(0, len(definitions), 1))
    plt.yticks(np.arange(0, len(definitions), 1))
    plt.colorbar(label='Similarity Score')
    plt.title(title)
    plt.xlabel('Definitions')
    plt.ylabel('Definitions')
    plt.show()

def main():
    # Parâmetros
    original_csv = 'samples/tests_smells.csv'
    sample_size = 24
    gpt_output_csv = 'sample_similarity_matrix_gpt.csv'
    tfidf_output_csv = 'sample_similarity_matrix_tfidf.csv'
    sample_output_csv = 'sample_definitions.csv'
    
    # Carregar o CSV original
    df = load_csv(original_csv)
    if df is None:
        return

    # Selecionar a coluna "Definition" e amostrar
    try:
        sample_df = df[['Definition']].sample(n=sample_size, random_state=42)
    except KeyError:
        print("A coluna 'Definition' não foi encontrada no CSV.")
        return

    # Obter as definições
    definitions = sample_df['Definition'].tolist()

    # Salvar a amostra em um novo CSV
    save_csv(sample_df, sample_output_csv)

    # Calcular a matriz de similaridade usando GPT
    similarity_matrix_gpt = calculate_similarity_matrix_gpt(definitions)
    save_csv(pd.DataFrame(similarity_matrix_gpt), gpt_output_csv)

    # Calcular a matriz de similaridade usando TF-IDF
    similarity_matrix_tfidf = calculate_similarity_matrix_tfidf(definitions)
    save_csv(pd.DataFrame(similarity_matrix_tfidf), tfidf_output_csv)

    # Obter os valores mínimo e máximo de ambas as matrizes para normalização
    min_value = min(np.min(similarity_matrix_gpt), np.min(similarity_matrix_tfidf))
    max_value = max(np.max(similarity_matrix_gpt), np.max(similarity_matrix_tfidf))

    # Visualizar a matriz de similaridade (GPT)
    plot_heatmap(similarity_matrix_gpt, 'Heatmap of Similarity Scores (GPT)', definitions, min_value, max_value)

    # Visualizar a matriz de similaridade (TF-IDF)
    plot_heatmap(similarity_matrix_tfidf, 'Heatmap of Similarity Scores (TF-IDF)', definitions, min_value, max_value)

if __name__ == "__main__":
    main()
