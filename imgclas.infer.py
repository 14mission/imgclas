#!/usr/bin/env python3
import re
from transformers import pipeline, ImageClassificationPipeline, AutoModelForImageClassification, AutoImageProcessor
from datasets import load_dataset, Image, ClassLabel
from PIL import Image as PILImage
from torch.nn.functional import sigmoid
import torch
import sys, os, re, glob

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
av = sys.argv
av.pop(0)
ac = 0
while ac < len(av):
  if av[ac][0] != '-' and csvfn == None: csvfn = av[ac]
  elif av[ac][0] != '-': raise Exception("second nonflag: "+av[ac])
  elif av[ac] == "-m": multilabel = True
  else: raise Exception("unkflag: "+av[ac])
  ac += 1

print("load csv")
dataset = load_dataset("csv", data_files=csvfn)["train"]
print(dataset[0])
print("verify images")
dataset = dataset.filter(lambda x: valid_image_path(x["image"]))
print("load images")
dataset = dataset.cast_column("image", Image())
print(dataset[0])
print("datasplit")
dataset_test = dataset.filter(lambda example, idx: idx % 10 == 9, with_indices=True)

modeldir = csvfn
modeldir = re.sub(r'\.csv','',csvfn)+".tagger/checkpoint-best"
print(f"modeldir={modeldir}")


class MultiLabelPipeline(ImageClassificationPipeline):
  def postprocess(self, model_outputs, **kwargs):
    logits = model_outputs["logits"][0]
    #print("logits="+",".join([str(l) for l in logits]))
    preds = [id2label[i] for i, l in enumerate(logits) if l > 0]
    return preds

print("load model")
#classifier = pipeline("image-classification", model=modeldir)
model = AutoModelForImageClassification.from_pretrained(modeldir)
processor = AutoImageProcessor.from_pretrained(modeldir)
classifier = MultiLabelPipeline(model=model,image_processor=processor) if multilabel else ImageClassificationPipeline(model=model,image_processor=processor)

print(classifier.model.config.id2label)
print(classifier.model.classifier) 

id2label = classifier.model.config.id2label
print("id2label: "+str(id2label))
print("foo:"+str(len(id2label)))
label2id = classifier.model.config.label2id
print("label2id: "+str(label2id))

print("cnv str labels to int")
if multilabel:
  def encode_multi_label(example):
    multi_hot = [1 if id2label[i] in example["label"].split(",") else 0 for i in range(len(id2label))]
    example["label"] = multi_hot
    return example
  dataset_test = dataset_test.map(encode_multi_label)
else:
  dataset_test = dataset_test.map(lambda ex: {"label": int(label2id[ex["label"]])})
  numperlabel = {}
  for lblnum in dataset_test["label"]:
    lblstr = id2label[lblnum]
    if lblstr not in numperlabel:
      numperlabel[lblstr] = 0
    numperlabel[lblstr] += 1
  print(", ".join(lbl+"="+str(numperlabel[lbl]) for lbl in sorted(numperlabel.keys())))

outfn = re.sub(r'\.csv','',csvfn)+".infer.out.tsv"
print(f"infer, write {outfn}")
outh = open(outfn,"w")
for i in range(len(dataset_test["image"])):
  imgdata = dataset_test["image"][i]
  reflabel = ",".join([id2label[j] for j, p in enumerate(dataset_test["label"][i]) if p > 0]) if multilabel else id2label[dataset_test["label"][i]]
  hyplabel = ",".join(classifier(imgdata)) if multilabel else classifier(imgdata)[0]["label"]
  print(
    re.sub(r'\s+','_',str(imgdata))
    +"\t"+reflabel
    +"\t"+hyplabel,
    file=outh)
