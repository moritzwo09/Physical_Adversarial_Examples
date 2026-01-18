#!/bin/bash

cd phys_imgs || exit
mkdir -p png

start=$(date +%s)

count=0
for f in *.HEIC; do
  new_name=$(printf "IMG_%02d.png" "$count")
  magick "$f" -colorspace sRGB -define png:bit-depth=8 "png/$new_name"
  echo "Finished $new_name"
  ((count++))
done

end=$(date +%s)
elapsed=$((end - start))

echo "Converted $count images in $elapsed seconds."