"""
This script fixes image path defined by versioning.
"""
import os
import tqdm
from bs4 import BeautifulSoup


def get_nb_img(tag, path):
    if not tag:
        return
    print(tag['src'])
    if 'grid00' in tag['src']:
        splited = tag['src'].split('/')
        abspath = os.path.join(*splited[splited.index('_build'):])
        tag['src'] = os.path.relpath(abspath, path)


for root, dir, files in os.walk('_build/html'):
    for file in files:
        path = os.path.join(root, file)
        if os.path.isfile(path) and os.path.splitext(path)[1] == '.html':
            with open(path) as reader:
                r = BeautifulSoup(reader.read())
                for tag in r.find_all('img'):
                    get_nb_img(tag, path)
            with open(path, 'w') as writer:
                writer.write(str(r))
