#!/bin/bash

cut -d'|' -f1 samples_urls.txt | sed 's|^https\?://||' | cut -d/ -f1 | sort | uniq -c | sort -n
