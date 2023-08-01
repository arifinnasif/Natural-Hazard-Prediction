import requests
from bs4 import BeautifulSoup as bs
import shutil
import os

cwd = os.getcwd()


_URL = "https://download.avl.class.noaa.gov/download/8324632223/001"

r = requests.get(_URL)
soup = bs(r.text, 'html.parser')
urls = []
names = []
for i, link in enumerate(soup.findAll('a')):
    _FULLURL = _URL + link.get('href')
    if _FULLURL.endswith('.nc'):
        urls.append(_FULLURL)
        names.append(soup.select('a')[i].attrs['href'])

names_urls = zip(names, urls)


download_dir = cwd+"\\glm\\"
for name, url in names_urls:
    r = requests.get(url, stream=True)
    with open(download_dir + name.split('/')[-1], "wb") as f: #last part ke file name dewar jonno
        shutil.copyfileobj(r.raw, f)



