import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_similarity_score_tfidf(definition1, definition2):
    print("Calculating similarity score using TF-IDF...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([definition1, definition2])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    similarity_score = similarity_matrix[0, 1]

    # Normalizando o score para uma escala de 0 a 100
    similarity_score = min(100, max(0, similarity_score * 100))
    similarity_score = round(similarity_score, 2) 
    print("Similarity score using TF-IDF:", similarity_score)
    return similarity_score

def calculate_similarity_matrix_tfidf(definitions):
    n = len(definitions)
    similarity_matrix = np.zeros((n, n))
    print("Calculating similarity matrix using TF-IDF...")
    for i in range(n):
        similarity_matrix[i, i] = 100.0
        for j in range(i + 1, n):
            print(f"Processing definitions {i + 1} and {j + 1}...")
            similarity = get_similarity_score_tfidf(definitions[i], definitions[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Matriz simétrica
    print("Similarity matrix calculation using TF-IDF complete.")
    return similarity_matrix
