# Use images on gemma model
# The image model is heavy weight so will check if my laptop with 16GB can handle it :)
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch

model_path = "C:\\Users\\paman\\OneDrive\\Desktop\\Python-workspace\\paligemma-3b-mix-224"

# Load the processor and model
processor = PaliGemmaProcessor.from_pretrained(model_path)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.float32,
    device_map=None,           # disables auto-mapping
    low_cpu_mem_usage=True     # prevents memory spikes
)
# Set device manually
device = "cpu"
model.to(device)

image_path = r"C:\Users\paman\OneDrive\Desktop\Python-workspace\nish_test_pic.jpg"
image = Image.open(image_path).convert("RGB")

# Example text prompt
prompt = "Describe this image."

# Prepare inputs
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

# Generate output
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=50)

# Decode the generated text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Prompt:", prompt)
print("Model output:", generated_text)