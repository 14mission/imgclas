#!/usr/bin/env python3
from datasets import load_dataset, Image, ClassLabel
from PIL import Image as PILImage
import sys, os, re, glob

def valid_image_path(path):
  try:
    with PILImage.open(path) as img:
      img.verify()  # quick integrity check
      return True
  except Exception:
    print(f"invalid image path {path}")
    return False

av = sys.argv
av.pop(0)
if len(av) != 1:
  raise Exception("usage: imgcla.train.py imglist.csv")
csvfn = av[0]

print("load csv")
dataset = load_dataset("csv", data_files=csvfn)["train"]
print(dataset[0])
print("verify images")
dataset = dataset.filter(lambda x: valid_image_path(x["image"]))
print("load images")
dataset = dataset.cast_column("image", Image())
print(dataset[0])
print("cnv labels")
labels = list(set(dataset["label"]))
dataset = dataset.cast_column("label", ClassLabel(names=labels))
print(dataset[0])

print("labelmaps")
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    print(f"{i} {label}")

print("numperlabel in train set")
numperlabel = {}
for item in dataset:
  lblstr = id2label[str(item["label"])]
  if lblstr not in numperlabel:
    numperlabel[lblstr] = 0
  numperlabel[lblstr] += 1

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
dataset = dataset.with_transform(transforms)

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

print("datasplit")
dataset_train = dataset.filter(lambda example, idx: idx % 10 < 8, with_indices=True)
dataset_vali = dataset.filter(lambda example, idx: idx % 10 == 8, with_indices=True)

modeldir = csvfn
modeldir = re.sub(r'\.csv','',csvfn)+".tagger"
print(f"modeldir={modeldir}")
for oldcheckpoint in glob.glob(modeldir+"/checkpoint-*"):
  print("del old: "+oldcheckpoint)
  shutil.rmtree(oldcheckpoint)

print("training")
training_args = TrainingArguments(
    output_dir=modeldir,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    #num_train_epochs=3,
    num_train_epochs=1,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_vali,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()

# copy best checkpoint to checkpoint-best
print("best checkpoint: "+trainer.state.best_model_checkpoint)
import shutil
shutil.copytree(trainer.state.best_model_checkpoint,modeldir+"/checkpoint-best")
