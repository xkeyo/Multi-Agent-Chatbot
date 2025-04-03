import requests
import json

def ask_ollama(prompt: str, model: str = 'llama3.2') -> str:
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={'model': model, 'prompt': prompt},
        stream=True
    )

    output = ''
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            chunk = data.get('response', '')
            output += chunk
    return output
