#!/usr/bin/env python3
import re
from datasets import load_dataset
from transformers import pipeline

print("load dataset")
food = load_dataset("food101", split="train[:5000]")
print("sample")
print(food[0])
food_test = food.filter(lambda example, idx: idx % 10 == 9, with_indices=True)

print("load model")
classifier = pipeline("image-classification", model="schmecktgut/checkpoint-best")
id2label = classifier.model.config.id2label

numperlabel = {}
for lblnum in food_test["label"]:
  lblstr = id2label[lblnum]
  if lblstr not in numperlabel:
    numperlabel[lblstr] = 0
  numperlabel[lblstr] += 1
print(", ".join(lbl+"="+str(numperlabel[lbl]) for lbl in sorted(numperlabel.keys())))

outfn = "food.infer.out.tsv"
print(f"infer, write {outfn}")
outh = open(outfn,"w")
for i in range(len(food_test["image"])):
  imgdata = food_test["image"][i]
  reflabel = id2label[food_test["label"][i]]
  results = classifier(imgdata)
  print(
    re.sub(r'\s+','_',str(imgdata))
    +"\t"+str(reflabel)
    +"\t"+str(results[0]["label"]),
    file=outh)
