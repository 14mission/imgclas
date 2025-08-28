#!/usr/bin/env python3
import re
from transformers import pipeline
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
  raise Exception("usage: imgclas.infer.py imglist.csv")
csvfn = av[0]

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

print("load model")
classifier = pipeline("image-classification", model=modeldir)
id2label = classifier.model.config.id2label
print("id2label: "+str(id2label))
label2id = classifier.model.config.label2id
print("label2id: "+str(label2id))

print("cnv str labels to int")
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
  reflabel = id2label[dataset_test["label"][i]]
  results = classifier(imgdata)
  print(
    re.sub(r'\s+','_',str(imgdata))
    +"\t"+str(reflabel)
    +"\t"+str(results[0]["label"]),
    file=outh)
