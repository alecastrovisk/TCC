from env import API_KEY
import requests
import time

count = 0

def get_similarity_score(definition1, definition2):
    global count

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    link = 'https://api.openai.com/v1/chat/completions'
    model_id = 'gpt-3.5-turbo'

    score_str = ''

    prompt = f"Classifique o nível de similaridade entre estes textos de 0-100 e se eles definem a mesma coisa(responda com o padrão  de output: [score: 100, 'sim']):\n\nTexto 1: {definition1}\n\nTexto 2: {definition2}"

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": f"{prompt}"}]
    }

    retries = 0
    max_retries = 5
    wait_time = 1

    while retries < max_retries:
        try:
            response = requests.post(link, headers=headers, json=body)
            if response.status_code == 429:
                print(f"Too many requests. Waiting {wait_time} seconds and retrying...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2
                continue
            
            response.raise_for_status()

            response_json = response.json()

            if 'choices' in response_json and len(response_json['choices']) > 0 and 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                score_str = response_json['choices'][0]['message']['content']
                
                if 'score' in score_str:
                    try:
                        score = float(score_str.split(',')[0].split(':')[1].strip())
                        count += 1
                    except (IndexError, ValueError):
                        print("Error parsing score_str. Check the format.")
                        score = -1
                else:
                    print(f"🚀 ~ score: {score_str}")
                    score = -1
            else:
                print("Unexpected response format.")
                score = -1

        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            score = -1
        except KeyError as e:
            print(f"Error processing response: {e}")
            score = -1
        except ValueError as e:
            print(f"Error converting score to float: {e}")
            score = -1

        print(f"score: {score}, \nresponse: {score_str}, \ncount: {count}")
        if score != -1:
            break

    return score
