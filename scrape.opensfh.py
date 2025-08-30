#!/usr/bin/env python3
import glob, re, os, sys

outh = open("opensfhistory.imgs.csv","w")

for dispfn in glob.glob("websites/opensfhistory.org/Display/*jpg"):
  print(f"read {dispfn}")
  content = None
  try:
    fh = open(dispfn)
    content = " ".join([ln.strip() for ln in fh])
  except:
    print(f"can´t read {dispfn}")
    continue

  # <meta name="description" content="Western Neighborhoods Project Image - Muralist Yana Zegri and son in front of Rainbow Mural about to be whitewashed by Revival of the Fittest. Mural was repainted and is still there as of 2021." >
  m = re.search(r'<title>.*<meta name="description"[^<>]*content="([^">]+)', content)
  description = None if m == None else m.group(1)
  print(f" description: {description}")
  if description == None:
    continue

  # <meta property="og:title" content="Cole & Haight" />
  m = re.search(r'<meta property="og:title" content="([^"<>]+)', content)
  title = None if m == None else m.group(1)
  print(f" title: {title}")
  if title == None:
    continue

  # <img src="/Image/800/wnp72.1285.jpg" class="img-responsive" alt="Powell & Market">
  m = re.search(r'<img[^<>]*src="(/Image\S+\.jpg)"',content)
  imgfile = None if m == None else m.group(1)
  if imgfile == None:
    print(" no img")
    continue

  # images may have moved!
  imgfile = re.sub(r'^.*\/','',imgfile)
  candidates = glob.glob("websites/opensfhistory.org/Image/*/"+imgfile)
  #print(" imgcands: "+", ".join(candidates))
  if len(candidates) == 0: 
    print("no img found")
    continue
  imgfile = candidates[-1]

  print(imgfile + "," + title, file=outh)
