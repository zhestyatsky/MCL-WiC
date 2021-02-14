#!/usr/bin/env bash

rm -rf data/
mkdir data/
git clone https://github.com/SapienzaNLP/mcl-wic.git data/ && unzip data/SemEval-2021_MCL-WiC_all-datasets.zip -d data/ && rm -rf data/*.{zip,md} data/.git data/MCL-WiC/*.txt
wget https://pilehvar.github.io/wic/package/WiC_dataset.zip -P data/ && unzip data/WiC_dataset.zip -d data/SuperGLUE-WiC/ && rm -r data/*.zip
