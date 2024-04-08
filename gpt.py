from env import API_KEY
import requests
import json

def get_similarity_score(definition1, definition2):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    link = 'https://api.openai.com/v1/chat/completions'
    model_id = 'gpt-3.5-turbo'

    prompt = f"Classifique o nível de similaridade entre estes textos de 0-100 e se eles definem a mesma coisa(responda com o padrão  de output: [score: 100, 'sim']):\n\nTexto 1: {definition1}\n\nTexto 2: {definition2}"

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": f"{prompt}"}]
    }

    body = json.dumps(body)

    req = requests.post(link, headers=headers, data=body)

    # print(req.text)

    response = req.json()
    score_str = response['choices'][0]['message']['content']
    print("🚀 ~ score:", score_str)
    try:
        score = int(score_str.split(',')[0].split(':')[1].strip())
        
    except ValueError:
        score = 0  
    print("🚀 ~ score:", score_str)
    return score


# result = get_similarity_score(definition1, definition2)
