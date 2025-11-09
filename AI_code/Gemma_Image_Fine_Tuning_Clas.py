# Now we will do fine tuning on the gemma lower image model
# pip install datasets transformers pandas pillow
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    Trainer,
    default_data_collator,
    TrainingArguments
)
from datasets import load_dataset, Dataset
from PIL import Image
import pandas as pd
import torch
import os

model_path = r"C:\Users\paman\OneDrive\Desktop\Python-workspace\paligemma-3b-mix-224"
csv_path = r"C:\Users\paman\OneDrive\Desktop\Python-workspace\family_dataset.csv"
image_folder = r"C:\Users\paman\OneDrive\Desktop\Python-workspace\Tejaspic"
output_dir = r"C:\Users\paman\OneDrive\Desktop\Python-workspace\paligemma_finetuned"
# ----------------------------
# LOAD MODEL + PROCESSOR
# ----------------------------
processor = PaliGemmaProcessor.from_pretrained(model_path)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True
)

device = "cpu"
model.to(device)
# ----------------------------
# LOAD DATASET
# CSV must have columns: image_path, caption

df = pd.read_csv(csv_path)
print (df.head())

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
# PREPROCESS
# ----------------------------
def preprocess_function(examples):
    images = [img for img in examples["image"] if img is not None]
    captions = [cap for (img, cap) in zip(examples["image"], examples["caption"]) if img is not None]

    if len(images) == 0:
        return {}

    # Add the <image> token so text and image align
    texts = [f"<image> {cap}" for cap in captions]

    inputs = processor(
        text=texts,
        images=images,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 👇 Set labels = input_ids (for language modeling)
    inputs["labels"] = inputs["input_ids"].clone()

    # Convert tensors to lists for HF Dataset compatibility
    batch = {k: v.tolist() for k, v in inputs.items()}
    return batch

processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names  # <- CRUCIAL
)

# ----------------------------
# TRAINING SETUP
# ----------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    save_strategy="epoch",
    report_to="none",  # optional, disables wandb/tensorboard
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
sample = processed_dataset[0]
for k, v in sample.items():
    print(k, type(v))

trainer.train()

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)