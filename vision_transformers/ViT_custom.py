#%%
import os
import glob
import pandas as pd
from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import tqdm

# %%
# Load dataset
train, val, id2label, label2id = VisionDataset.fromImageFolder(
    "./train/",
    test_ratio=0.1,
    balanced=True,
    augmentation=True,
    torch_vision=False
)

# %%
# Model setup
huggingface_model = 'google/vit-base-patch16-224-in21k'

# Create trainer
trainer = VisionClassifierTrainer(
    model_name='MyDogClassifier',
    train=train,
    test=val,
    output_dir="./out/",
    max_epochs=20,
    batch_size=4,
    lr=2e-5,
    fp16=False,  # Disable FP16 training for CPU
    model=ViTForImageClassification.from_pretrained(
        huggingface_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    ),
    feature_extractor=ViTFeatureExtractor.from_pretrained(huggingface_model),
)

# %%
# Force CPU execution
device = torch.device("cpu")
trainer.model.to(device)

# Run through test set to ensure CPU compatibility (optional)
for image, label in tqdm.tqdm(trainer.test):
    inputs = trainer.feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = trainer.model(**inputs)

# %%
# Evaluate F1 score
y_true, y_pred = trainer.evaluate_f1_score()

# %%
# Confusion Matrix
labels = list(label2id.keys())
cm = confusion_matrix(y_true, y_pred, labels=[label2id[label] for label in labels])
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", annot_kws={'size': 8})
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("./conf_matrix_1.jpg")

# %%
# %%
# Initialize classifier with proper label maps (fixed)
# %%
# Fix: Manually set id2label and label2id before wrapping in inference class
from transformers import ViTConfig

model = ViTForImageClassification.from_pretrained(huggingface_model)
model.config.id2label = {int(k): v for k, v in id2label.items()}
model.config.label2id = {v: int(k) for k, v in id2label.items()}  # Reverse of id2label

classifier = VisionClassifierInference(
    feature_extractor=ViTFeatureExtractor.from_pretrained(huggingface_model),
    model=model
)
# %%
# Inference for a single image
# Inference for a single image
test_img = "./test/affenpinscher/affenpinscher_0.jpg"
predicted_label = classifier.predict(img_path=test_img)

print(f"Image: {test_img}")
print(f"Predicted label: {predicted_label}")

# %%
# Batch inference for all test images
test_files = [f for f in glob.glob("./test/**/**", recursive=True) if os.path.isfile(f)]

for img_path in test_files:
    label = classifier.predict(img_path=img_path)
    print(f"{img_path} --> Predicted: {label}")
# %%
