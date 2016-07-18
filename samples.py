#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to download, cache and check samples
"""

from __future__ import print_function
import requests
import os
import hashlib
import base64


def encode_key(s):
    return base64.b16encode(hashlib.sha1(s.encode('utf-8')).digest()).lower()


def cache_store(url, content):
    key = encode_key(url)
    with open(os.path.join(b'.cache', key), 'w') as f:
        f.write(content)


def cache_get(url):
    key = encode_key(url)
    try:
        with open(os.path.join(b'.cache', key)) as f:
            return f.read()
    except IOError:
        return None


def download_url(url):
    from_cache = cache_get(url)
    if from_cache:
        print('Using from cache %s' % url)
        return from_cache
    else:
        result = requests.get(url).text
        print('Downloaded %s' % url)
        cache_store(url, result)
        return result


def get_dataset():
    with open('samples_urls.txt') as f:
        url_samples = [l.split('|') for l in f.readlines()]

    pages = [
        dict(url=url, page=download_url(url), target=target.strip())
        for url, target in url_samples
    ]
    print('Total page samples: %d' % len(pages))

    return pages


def run(args):
    get_dataset()


if '__main__' == __name__:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    args = parser.parse_args()
    run(args)
