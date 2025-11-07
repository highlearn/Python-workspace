
from ollama import Client

MODEL = "gemma3"
client = Client(host="http://localhost:11434")
prompt = input ("Enter your prompt: ")
print(f"\n--- Generating response from {MODEL} model ---\n")
res= client.generate(model=MODEL, prompt=prompt, max_tokens=512)
print(res.text)