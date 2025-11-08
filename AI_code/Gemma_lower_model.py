# We will create a response from Gemma model 
# pip install torch transformers

# using the Hugging Face Transformers library 
# or the Keras library. These libraries handle 
# the model loading and inference locally within 
# your Python environment, given you have the 
# necessary hardware (typically a GPU with sufficient VRAM
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
# # model_name = "sshleifer/tiny-gpt2"

model_path = "C:\\Users\\paman\\OneDrive\\Desktop\\Python-workspace\\gemma-3-1b-it"

dtype = torch.float16 
tokenizer = AutoTokenizer.from_pretrained(model_path)
# print (tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto", # Automatically distributes the model across available devices (GPU/CPU)
    dtype=dtype
    )       

messages = [
    {"role": "user", "content": "how is weather in india"}
]


prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
prompt += "<start_of_turn>model\n" 

# Encode the prompt and move to the model's device
inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

# # Generate the response
with torch.no_grad():
    outputs = model.generate(
        inputs, 
        max_new_tokens=200,     # Maximum number of tokens to generate
        do_sample=True,         # Enable sampling
        temperature=0.7,        # Sampling temperature
        top_k=50,               # Top-k sampling
        top_p=0.95              # Top-p (nucleus) sampling
    )

# Decode and print only the generated text (exclude the input prompt)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)