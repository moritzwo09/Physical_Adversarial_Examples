#!/bin/bash

cd phys_imgs || exit
mkdir -p png

count=0
for f in *.HEIC; do
  new_name=$(printf "IMG_%d.png" "$count")
  magick "$f" -colorspace sRGB -define png:bit-depth=8 "png/$new_name"
  ((count++))
done