#!/usr/bin/env python3
import glob, re, os, sys

dataset = []

for dispfn in glob.glob("websites/opensfhistory.org/Display/*jpg"):
  #print(f"read {dispfn}")
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
  #print(f" description: {description}")
  if description == None:
    print(f"no description in {dispfn}")
    continue

  # <meta property="og:" content="Cole & Haight" />
  m = re.search(r'<meta property="og:title" content="([^"<>]+)', content)
  titlestr = None if m == None else m.group(1)
  #print(f" title: {titlestr}")
  if titlestr == None:
    print(f"no title in {dispfn}")
    continue

  # <meta name="keywords" content="people posed, San Francisco History, San Francisco" >
  m = re.search(r'<meta name="keywords"[^<>]*content="([^"<>]+)', content)
  keywordstr = None if m == None else m.group(1)
  #print(f" keywords: {keywordstr}")
  if keywordstr == None:
    print(f"no keywords in {dispfn}")
    continue

  # <img src="/Image/800/wnp72.1285.jpg" class="img-responsive" alt="Powell & Market">
  m = re.search(r'<img[^<>]*src="(/Image\S+\.jpg)"',content)
  imgfile = None if m == None else m.group(1)
  if imgfile == None:
    print(f"no img in {dispfn}")
    continue

  # images may have moved!
  imgfile = re.sub(r'^.*\/','',imgfile)
  candidates = glob.glob("websites/opensfhistory.org/Image/*/"+imgfile)
  #print(" imgcands: "+", ".join(candidates))
  if len(candidates) == 0: 
    print(f"no img found for {dispfn}")
    continue
  imgfile = candidates[-1]

  dataset.append({"img": imgfile, "title":titlestr, "keywords":keywordstr, "description":description})

print("num items: "+str(len(dataset)))

def norm(s):
  s = s.lower()
  s = re.sub(r'\[.*?\]',' ',s)
  s = re.sub(r'\b(western neighborhoods)(\s*project)?(\s*image)?',' ',s)
  s = re.sub(r'\b(san[\s_]francisco|history)\b',' ',s)
  s = " ".join(re.findall(r'(\w+(?:\'\w+)*)',s) )
  s = re.sub(
    r'\b(golden gate park|golden gate|market street railway|market street|cliff house|southern pacific|key system)\b', # data-driven set would be better!
    lambda m: "_".join(m.group(1).split()), s)
  return s

#top words
print("wfreq")
wfreq = {}
bgfreq = {}
tgfreq = {}
for item in dataset:
  for s in [item["title"],item["keywords"],item["description"]]:
    s = norm(s)
    toks = s.split()
    for i in range(len(toks)):
      w = toks[i]
      if w not in wfreq:
        wfreq[w] = 0 
      wfreq[w] += 1
      if i+1 < len(toks):
        bg = toks[i]+" "+toks[i+1]
        if bg not in bgfreq:
          bgfreq[bg] = 0
        bgfreq[bg] += 1
      if i+2 < len(toks):
        tg = toks[i]+" "+toks[i+1]+" "+toks[i+2]
        if tg not in tgfreq:
          tgfreq[tg] = 0
        tgfreq[tg] += 1
for nw in [1,2,3]:
  print(f"{nw}grams")
  countset = wfreq if nw == 1 else ( bgfreq if nw == 2 else tgfreq )
  i = 0
  for wordandcount in sorted(countset.items(), key=lambda item: str(1.0/item[1])+item[0]):
    w, n = wordandcount
    print(f"{w}\t{n}")
    i += 1
    if i >= 25:
      break

excludewords = "the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us".split()

outfn = "opensfhistory.imgs.csv"
print(f"write {outfn}")
outh = open(outfn,"w")
for item in dataset:
  itemkw = {}
  for s in [item["title"],item["keywords"],item["description"]]:
    for w in norm(s).split():
      if w not in excludewords and len(w) > 1 and re.match(r'^.*[a-z]',w) != None:
        itemkw[w] = True
  print(item["img"]+",\""+",".join(sorted(itemkw.keys()))+"\"", file=outh)

print("done")
