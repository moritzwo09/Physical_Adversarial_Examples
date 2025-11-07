import cv2
from pathlib import Path
from groundingdino.util.inference import load_model, load_image, predict, annotate

# Verzeichnisse
INPUT_DIR = Path("/phys_imgs/png/")
OUTPUT_DIR = Path("/output/")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# GroundingDINO-Modell laden
model = load_model(
    "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "../GroundingDINO/weights/groundingdino_swint_ogc.pth"
)

# Parameter
TEXT_PROMPT = "only the Photography."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Startnummer
count = 9566

for img_path in sorted(INPUT_DIR.glob("*.png")):
    print(f"Processing {img_path.name} ...")

    # Bild laden
    image_source, image = load_image(str(img_path))

    # Vorhersage
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # Bounding Box zeichnen
    annotated_frame = annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=phrases
    )

    # Ausgabe speichern
    output_path = OUTPUT_DIR / f"IMG_{count:04d}_bb.jpg"
    cv2.imwrite(str(output_path), annotated_frame)

    count += 1

print("Done.")