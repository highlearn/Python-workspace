# ---------------------------------------------------------
# Fine-tuning PaliGemma on custom image-caption dataset
# ---------------------------------------------------------
# pip install datasets transformers pandas pillow torch

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    Trainer,
    default_data_collator,
    TrainingArguments
)
from datasets import Dataset
from PIL import Image
import pandas as pd
import torch
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ----------------------------
# PATHS
# ----------------------------
model_path = r"C:\Users\paman\OneDrive\Desktop\Python-workspace\paligemma-3b-mix-224"
csv_path = r"C:\Users\paman\OneDrive\Desktop\Python-workspace\family_dataset.csv"
image_folder = r"C:\Users\paman\OneDrive\Desktop\Python-workspace\Tejaspic"
output_dir = r"C:\Users\paman\OneDrive\Desktop\Python-workspace\paligemma_finetuned"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# LOAD MODEL + PROCESSOR
# ----------------------------
print("🔹 Loading model and processor...")
processor = PaliGemmaProcessor.from_pretrained(model_path)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded successfully on {device.upper()}.")

# ----------------------------
# LOAD DATASET
# ----------------------------
df = pd.read_csv(csv_path)
print("\n✅ Loaded CSV:")
print(df.head())

def load_image(example):
    image_path = os.path.join(image_folder, example["image_path"])
    if not os.path.exists(image_path):
        print(f"⚠️ File not found: {image_path}")
        example["image"] = None
        return example
    image = Image.open(image_path).convert("RGB")
    example["image"] = image
    return example

dataset = Dataset.from_pandas(df)
dataset = dataset.map(load_image)

# ----------------------------
# PREPROCESS FUNCTION
# ----------------------------
def preprocess_function(examples):
    images, captions = [], []
    for img, cap in zip(examples["image"], examples["caption"]):
        if img is not None:
            images.append(img)
            captions.append(cap)

    if len(images) == 0:
        return {}

    # Each caption should correspond to one image, with one <image> token
    texts = [f"<image>{cap}" for cap in captions]

    # No truncation, let model handle input length
    inputs = processor(
    text=texts,
    images=images,
    padding="longest",   # or padding=False
    truncation=False,    # don’t cut text
    return_tensors="pt"
)

    inputs["labels"] = inputs["input_ids"].clone()
    batch = {k: v.tolist() for k, v in inputs.items()}
    return batch

processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

print(f"\n✅ Processed dataset size: {len(processed_dataset)}")
print("Columns:", processed_dataset.column_names)
print("Sample keys and types:")
for k, v in processed_dataset[0].items():
    print(f"  {k} {type(v)}")

# ----------------------------
# TRAINING SETUP
# ----------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="epoch",          # ensures saving even for small datasets
    save_total_limit=1,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=default_data_collator,
)

# ----------------------------
# TRAIN
# ----------------------------
print("\n🚀 Starting training...")
trainer.train()
print("✅ Training complete!")

# ----------------------------
# SAVE MODEL + PROCESSOR
# ----------------------------
print("\n💾 Saving model and processor...")
trainer.save_model(output_dir)             # ensures model + tokenizer saved
processor.save_pretrained(output_dir)
print(f"✅ Model and processor saved successfully to:\n   {output_dir}")