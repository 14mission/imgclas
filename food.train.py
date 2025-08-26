#!/usr/bin/env python3
from datasets import load_dataset

print("load dataset")
food = load_dataset("food101", split="train[:5000]")
print("set split")
food = food.train_test_split(test_size=0.2)
print("sample")
print(food["train"][0])

print("labelmaps")
labels = food["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    if i < 5:
        print(f"{i} {label}")

# load vit model
# (we need it first for turning images into features)
print("load vit model")
from transformers import AutoImageProcessor
checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint,use_fast=True)

# apply some image transformations
print("image transforms")
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples
food = food.with_transform(transforms)

print("create batch of examples")
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()

print("set up eval")
import evaluate
accuracy = evaluate.load("accuracy")
import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

print("really load model")
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

print("training")
modeldir = "schmecktgut"
training_args = TrainingArguments(
    output_dir=modeldir,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    #num_train_epochs=1,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=food["train"],
    eval_dataset=food["test"],
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()

# copy best checkpoint to checkpoint-best
print("best checkpoint: "+trainer.state.best_model_checkpoint)
import shutil
shutil.copytree(trainer.state.best_model_checkpoint,modeldir+"/checkpoint-best")

