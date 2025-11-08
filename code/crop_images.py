import cv2
from pathlib import Path
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
import time


################################## get bounding boxes #################################
# Verzeichnisse
INPUT_DIR = Path("./phys_imgs/png/")
OUTPUT_DIR = Path("./output/bb/")
CROPPED_OUTPUT_DIR = Path("./output/crops/")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# GroundingDINO-Modell laden
model = load_model(
    "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "../GroundingDINO/weights/groundingdino_swint_ogc.pth"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def get_boxes():
    # Parameter
    prompt = "only the Paper with the foto on it."
    box_threshold = 0.4
    text_threshold = 0.25

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
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cpu",
        )

        # Bounding Box zeichnen
        annotated_frame = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases
        )

        # Ausgabe speichern
        output_path = OUTPUT_DIR / f"IMG_{count}_bb.jpg"
        cv2.imwrite(str(output_path), annotated_frame)

        # Beispiel: eine Box aus predict()
        box = boxes[0]  # [cx, cy, w, h]
        cx, cy, w, h = box.tolist()

        # Größe des Bildes ermitteln
        height, width = image_source.shape[:2]

        # relative Werte in absolute Pixel umrechnen
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)

        # auf gültige Bereiche begrenzen
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width), min(y2, height)

        # Bild zuschneiden
        crop = image_source[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{CROPPED_OUTPUT_DIR}/cropped_{count}.jpg", crop)

        count += 1

    print("Done.")

before = time.time()
get_boxes()
after = time.time()
time_elapsed = after - before
print(f"Creating bounding boxes took: {time_elapsed:.2f} seconds.")

################################## cut out traffic signs ###############################
