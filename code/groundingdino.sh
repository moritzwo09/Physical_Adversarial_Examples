#!/bin/bash

# Verzeichnisse
INPUT_DIR="../code/phys_imgs/png/"
OUTPUT_DIR="../code/output/"

# Nummer counter
COUNT=9566

# In das GroundingDINO-Verzeichnis wechseln
cd ../GroundingDINO || exit

# Über alle PNG-Bilder im Input-Ordner iterieren
for IMG_PATH in "$INPUT_DIR"*.png; do
    echo "Processing $IMG_PATH ..."

    # GroundingDINO ausführen
    PYTHONWARNINGS="ignore" python ./demo/inference_on_a_image.py \
    -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
    -p weights/groundingdino_swint_ogc.pth \
    -i "$IMG_PATH" \
    -o "$OUTPUT_DIR" \
    -t "only the Fotography" \
    --cpu-only > /dev/null 2>&1

    # raw_image.jpg löschen, falls vorhanden
    if [ -f "$OUTPUT_DIR/raw_image.jpg" ]; then
        rm "$OUTPUT_DIR/raw_image.jpg"
    fi

    # pred.jpg umbenennen
    if [ -f "$OUTPUT_DIR/pred.jpg" ]; then
        mv "$OUTPUT_DIR/pred.jpg" "$OUTPUT_DIR/IMG_$(printf "%04d" $COUNT)_bb.jpg"
    fi
    # Counter erhöhen
    COUNT=$((COUNT+1))
done

echo "Done."