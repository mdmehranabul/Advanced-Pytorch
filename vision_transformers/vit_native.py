# vit_native.py

from datasets import load_dataset
from transformers import ViTForImageClassification, AutoImageProcessor, TrainingArguments, Trainer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import numpy as np
import evaluate
import os

# Config
MODEL_NAME = "google/vit-base-patch16-224-in21k"
DATA_DIR = "./train"  # folder containing subfolders of classes
OUTPUT_DIR = "./vit_output"
NUM_EPOCHS = 5
BATCH_SIZE = 4

# Load dataset from folders
dataset = load_dataset("imagefolder", data_dir=DATA_DIR, split="train").train_test_split(test_size=0.1, seed=42)

# Extract label info
label2id = {label: idx for idx, label in enumerate(dataset['train'].features['label'].names)}
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

# Load processor
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# Define transforms
transform = Compose([
    Resize((224, 224)),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std),
])

def transform_fn(example):
    example['pixel_values'] = transform(example['image'])
    return example

# Apply transforms
dataset = dataset.map(transform_fn, batched=False)
dataset.set_format(type="torch", columns=["pixel_values", "label"])

# Load model
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Metric
accuracy = evaluate.load("accuracy")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=p.label_ids)

# Training args
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    learning_rate=2e-5,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model & processor
model.save_pretrained(os.path.join(OUTPUT_DIR, "model"))
processor.save_pretrained(os.path.join(OUTPUT_DIR, "processor"))
