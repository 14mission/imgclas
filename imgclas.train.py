#!/usr/bin/env python3
from datasets import load_dataset, Image, ClassLabel
from PIL import Image as PILImage
import sys, os, re, glob, shutil

def valid_image_path(path):
  try:
    with PILImage.open(path) as img:
      img.load()
    return True
  except Exception:
    print(f"invalid image path {path}")
    return False

csvfn = None
multilabel = False
num_train_epochs = 1
learning_rate = 5e-5
av = sys.argv
av.pop(0)
ac = 0
while ac < len(av):
  if av[ac][0] != '-' and csvfn == None: csvfn = av[ac]
  elif av[ac][0] != '-': raise Exception("extra nonflag: "+av[ac])
  elif av[ac] == "-m": multilabel = True
  elif ac+1 == len(av) or av[ac+1][0] == '-': raise Exception("novalfor: "+av[ac])
  elif av[ac] == "-e": ac+=1; num_train_epochs = int(av[ac])
  elif av[ac] == "-l": ac+=1; learning_rate = float(av[ac])
  else: raise Exception("unkflag: "+av[ac])
  ac += 1
if csvfn == None:
  raise Exception("usage: imgclas.train.py imglist.csv")

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
if multilabel:
  splitlabels = set()
  for lbl_str in labels:
    split_labels = lbl_str.split(",")
    splitlabels.update(split_labels)
  labels = list(splitlabels)
  print("num uniq split labels: "+str(len(labels)))
  def encode_multi_label(example):
    multi_hot = [1 if lbl in example["label"].split(",") else 0 for lbl in labels]
    example["label"] = multi_hot
    return example
  dataset = dataset.map(encode_multi_label)
else:
  dataset = dataset.cast_column("label", ClassLabel(names=labels))
print(dataset[0])

print("labelmaps")
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
    if i < 20:
        print(f"{i} {label}")
if len(label2id.keys()) > 20:
    print("...")

#print("numperlabel in train set")
#numperlabel = {}
#for item in dataset:
#  lblstrs = [id2label[str(item["label"])]]
#  for libstr in libstrs:
#    if lblstr not in numperlabel:
#      numperlabel[lblstr] = 0
#    numperlabel[lblstr] += 1

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

print("datasplit")
dataset_train = dataset.filter(lambda example, idx: idx % 10 < 8, with_indices=True)
dataset_vali = dataset.filter(lambda example, idx: idx % 10 == 8, with_indices=True)

print("create batch of examples")
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()

print("set up eval")
import evaluate
accuracy = evaluate.load("accuracy")
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
def compute_metrics(eval_pred):
    print("calling compute_metrics")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
def compute_metrics_multilabel(eval_pred):
    print("calling compute_metrics_multilabel")
    logits, labels = eval_pred
    print("logits[0]"+",".join([str(x) for x in logits[0:10]]))

    print("test low thresh")
    lowthreshpreds = (logits > -0.8).astype(int)
    print("lowthreshf1="+str(f1_score(labels, lowthreshpreds, average="micro"))
      + "; prec="+str(precision_score(labels, lowthreshpreds, average="micro"))
      + "; rec="+str(recall_score(labels, lowthreshpreds, average="micro"))
    )

    preds = (logits > 0).astype(int)   # threshold at 0
    return {
        "f1": f1_score(labels, preds, average="micro"),
        "precision": precision_score(labels, preds, average="micro"),
        "recall": recall_score(labels, preds, average="micro"),
    } 

#pos weight function for imbalanced multilabel
from transformers import Trainer
if multilabel:
    import numpy as np
    import torch
    print("fnoop: "+str(dataset_train[0]["label"]))
    labels_list = [row["label"] for row in dataset_train]
    labels_np = np.array(labels_list, dtype=np.float32)
    print(str(labels_np))
    pos_counts = labels_np.sum(axis=0)
    neg_counts = len(labels_np) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-5), dtype=torch.float)
    print("pos_weight="+str(pos_weight))

    # custom Trainer
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
    
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits, labels.type_as(logits))
    
            return (loss, outputs) if return_outputs else loss
else:
    class CustomTrainer(Trainer):
        def bogusfunc():
            print("foo")

print("really load model")
from transformers import AutoModelForImageClassification, TrainingArguments
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    problem_type=("multi_label_classification" if multilabel else "single_label_classification"),
)

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
    #for quick debug runs
    #eval_strategy="steps",
    #save_strategy="steps",
    #save_steps = 1, # was 25
    #eval_steps = 1, # was 25
    learning_rate=learning_rate,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model=("eval_f1" if multilabel else "accuracy"),
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_vali,
    processing_class=image_processor,
    compute_metrics=(compute_metrics_multilabel if multilabel else compute_metrics),
)

trainer.train()

# copy best checkpoint to checkpoint-best
print("best checkpoint: "+trainer.state.best_model_checkpoint)
if not os.path.isdir(trainer.state.best_model_checkpoint):
    raise Exception("no dir: "++trainer.state.best_model_checkpoint)
import shutil
shutil.copytree(trainer.state.best_model_checkpoint,modeldir+"/checkpoint-best")
