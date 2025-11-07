# install the Ollama package for python
# pip install ollama
# ollama pull gemma3

from ollama import chat

response = chat(
        model='gemma3',  # Specify the Gemma 3 model you pulled
        messages=[
            {
                'role': 'user',
                'content': 'Why Indian names have meanings?'
            }
        ]
    )
print(response['message']['content'])

# print(response)

# This code will fetch a response from the Gemma 3 model using Ollama's Python package.