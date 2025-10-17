#!/bin/bash

echo "Downloading model"
uv add gdown
gdown 1qhasZiyNAUU7xGDFilAlC_KTwKEL5mHk
unzip rtdetr-v2-r18.zip
rm -rf __MACOSX
rm -f rtdetr-v2-r18.zip
