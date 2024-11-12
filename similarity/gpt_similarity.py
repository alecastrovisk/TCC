import requests
import time
import numpy as np
from env import API_KEY

count = 0

def get_similarity_score_gpt(definition1, definition2):
    global count

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    link = 'https://api.openai.com/v1/chat/completions'
    model_id = 'gpt-3.5-turbo'

    prompt = f"Classifique o nível de similaridade entre estes textos de 0-100 e se eles definem a mesma coisa(responda com o padrão de output: [score: 100, 'sim']):\n\nTexto 1: {definition1}\n\nTexto 2: {definition2}"

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}]
    }

    max_retries = 5
    wait_time = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(link, headers=headers, json=body)
            response.raise_for_status()
            
            if response.status_code == 429:
                print(f"Too many requests. Waiting {wait_time} seconds and retrying...")
                time.sleep(wait_time)
                wait_time *= 2
                continue

            response_json = response.json()
            score_str = response_json['choices'][0]['message']['content']

            if 'score' in score_str:
                try:
                    score = float(score_str.split(',')[0].split(':')[1].strip())
                    score = min(100, max(0, score))
                    print("Similarity score using GPT:", score) 
                    count += 1
                    return score
                except (IndexError, ValueError):
                    print("Error parsing score_str. Check the format.")
                    return -1
            else:
                print(f"Unexpected score format: {score_str}")
                return -1

        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                return -1
        except KeyError as e:
            print(f"Error processing response: {e}")
            return -1
        except ValueError as e:
            print(f"Error converting score to float: {e}")
            return -1

    return -1

def calculate_similarity_matrix_gpt(definitions):
    n = len(definitions)
    similarity_matrix = np.zeros((n, n))
    print("Calculating similarity matrix using GPT...")
    for i in range(n):
        similarity_matrix[i, i] = 100.0
        for j in range(i + 1, n):
            print(f"Processing definitions {i + 1} and {j + 1}...")
            similarity = get_similarity_score_gpt(definitions[i], definitions[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Matriz simétrica
    print("Similarity matrix calculation using GPT complete.")
    return similarity_matrix
