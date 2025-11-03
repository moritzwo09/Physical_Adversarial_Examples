#!/bin/bash

cd phys_imgs
mkdir png
for f in *.HEIC; do
  magick "$f" -colorspace sRGB -define png:bit-depth=8 "png/${f%.*}.png"
done