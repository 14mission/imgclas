#!/usr/bin/env python3
import re
import urllib.request

baseurl = "https://www.gearedsteam.com/"
urls = [baseurl]

didbefore = {}

def urlnorm(url):
  url = re.sub(r'#.*$','',url)
  while re.search(r'[^/]+/\.\./',url):
    url = re.sub(r'[^/]+/\.\./','',url)
  if re.search(r'/\.\.',url):
    raise Exception("relpath: "+url)
  url = re.sub(r'(\w)\/+(\w)/',r'\1/\2/',url)
  return url

def main():

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
      else:
        print("getimg: "+url+" -> "+img)
        didbefore[img] = True

main()
