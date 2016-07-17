#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to download, cache and check samples
"""

from __future__ import print_function
import requests
import pickle


SAMPLES_CACHE_PATH = 'cache.pickle'


try:
    with open(SAMPLES_CACHE_PATH) as f:
        SAMPLES_CACHE = pickle.load(f)
except:
    SAMPLES_CACHE = {}


def download_urls(urls):
    for url in urls:
        if url in SAMPLES_CACHE:
            print('Using from cache %s' % url)
            yield SAMPLES_CACHE[url]
        else:
            SAMPLES_CACHE[url] = requests.get(url).text
            print('Downloaded %s' % url)
            yield SAMPLES_CACHE[url]

    with open(SAMPLES_CACHE_PATH, 'w') as f:
        pickle.dump(SAMPLES_CACHE, f)


def get_dataset():
    with open('samples_urls.txt') as f:
        URL_SAMPLES = [l.split('|') for l in f.readlines()]

    pages = list(download_urls([url for url, _ in URL_SAMPLES]))
    print('Total page samples: %d' % len(pages))

    return pages


def run(args):
    get_dataset()


if '__main__' == __name__:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    args = parser.parse_args()
    run(args)
