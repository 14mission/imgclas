#!/usr/bin/env python3
import re
import urllib.request

def urlnorm(url):
  url = re.sub(r'#.*$','',url)
  while re.search(r'[^/]+/\.\./',url):
    url = re.sub(r'[^/]+/\.\./','',url)
  if re.search(r'/\.\.',url):
    raise Exception("relpath: "+url)
  url = re.sub(r'(\w)\/+(\w)/',r'\1/\2/',url)
  return url

def main():

  baseurl = "https://www.gearedsteam.com/"
  urls = [baseurl]
  didbefore = {}
  locotypes = {}

  outlist = open("gslw.tn.imgs.csv","w")

  while(len(urls)):
    url = urls.pop(0)
    print("get: "+url)
    print("stacksize: "+str(len(urls)))
    content = str(urllib.request.urlopen(url).read())
    for nexturl in re.findall(r'href="([^"]+)',content):
      if not re.match(r'^https?:',nexturl):
        nexturl = re.sub(r'[^\/]+$','',url)+nexturl
      nexturl = urlnorm(nexturl)
      print("next: "+nexturl)
      if not re.search(r'\.html?$',nexturl):
        print("notapage")
      elif re.search(r'[^\w:/\.-]',nexturl):
        print("badchar")
      elif len(nexturl) < len(baseurl) or nexturl[0:len(baseurl)] != baseurl:
        print("skipexternal")
      elif re.search(r'(\?|=|new|video|help|links|support|article|copyright|advert)',nexturl):
        print("skipquery")
      elif nexturl in didbefore:
        print("didbefore")
      else:
        print("APPEND")
        didbefore[nexturl] = True
        urls.append(nexturl)
    for img in re.findall(r'img [^>]*src="([^"]+)',content):
      if not re.match(r'^https?:',img):
        img = re.sub(r'[^\/]+$','',url)+img
      img = urlnorm(img)
      if img in didbefore:
        print("didimgbefore: "+img)
      elif re.search(r'[^\w:/\.-]',img):
        print("badchar: "+img)
      else:
        typematch = re.match(r'^.+\b(heisler|shay|climax|bell|dunkirk|baldwin|willamette|davenport|dewey|rod|byers|other|books)\b',img)
        locotype = ("unknown" if typematch == None else typematch.group(1))
        if locotype == "books" or locotype == "unknown":
          print("skiplocotype: "+locotype)
        else:
          print("getimg: "+url+" ("+locotype+") -> "+img)
          didbefore[img] = True
          localfn = "gslw.tn.imgs/" + re.sub(r'^.*/','',img)
          print(localfn+","+locotype,file=outlist)
          imgcontent = urllib.request.urlopen(img).read()
          imgout = open(localfn,"wb")
          imgout.write(imgcontent)
          if locotype not in locotypes:
            locotypes[locotype] = 0
          locotypes[locotype] += 1

  for locotype in sorted(locotypes.keys()):
    print(locotype+"="+str(locotypes[locotype]))

main()
