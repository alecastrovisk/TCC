from env import API_KEY
import requests


count = 0
def get_similarity_score(definition1, definition2):
    global count
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    link = 'https://api.openai.com/v1/chat/completions'
    model_id = 'gpt-3.5-turbo'

    prompt = f"Classifique o nível de similaridade entre estes textos de 0-100 e se eles definem a mesma coisa(responda com o padrão  de output: [score: 100, 'sim']):\n\nTexto 1: {definition1}\n\nTexto 2: {definition2}"

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": f"{prompt}"}]
    }

    try:
        response = requests.post(link, headers=headers, json=body)
        response.raise_for_status()  # Raise an exception for bad status codes
        response_json = response.json()
        score_str = response_json['choices'][0]['message']['content']
        if 'score' in score_str:
            score = int(score_str.split(',')[0].split(':')[1].strip())
            count += 1
        else:
            print(f"🚀 ~ score: {score_str}")
            score = 0
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        score = 0
    except KeyError as e:
        print(f"Error processing response: {e}")
        score = 0

    print(f"score:{ score}, \n response: {score_str}, \n count: {count}")

    return score

