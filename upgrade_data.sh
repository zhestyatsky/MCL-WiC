#!/usr/bin/env bash

rm -rf data/
mkdir data/
git clone https://github.com/SapienzaNLP/mcl-wic.git data/ && unzip data/SemEval-2021_MCL-WiC_all-datasets.zip -d data/ && unzip data/SemEval-2021_MCL-WiC_test-gold-data.zip -d data && rm -rf data/*.{zip,md} data/.git data/MCL-WiC/*.txt && mv data/test.en-*.gold data/MCL-WiC/test/crosslingual/ && mv data/*.gold data/MCL-WiC/test/multilingual/ && mv data/MCL-WiC/test/crosslingual/test.en-en.gold data/MCL-WiC/test/multilingual/
wget https://pilehvar.github.io/wic/package/WiC_dataset.zip -P data/ && unzip data/WiC_dataset.zip -d data/SuperGLUE-WiC/ && rm -r data/*.zip
