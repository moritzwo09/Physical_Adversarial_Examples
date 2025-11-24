#!/usr/bin/env python

import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import glob
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

png_folder = "digital_imgs"
png_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))

all_masks = []
all_scores = []
all_inputs = []

for path in png_files:
    img = Image.open(path).convert("RGB")
    w, h = img.size

    box = [[
        [int(0.1 * w), int(0.1 * h)],  # top-left
        [int(0.9 * w), int(0.9 * h)]  # bottom-right
    ]]

    inputs = processor(
        img,
        input_boxes=box,
        return_tensors="pt"
    )

    cpu_inputs = {k: v.clone() for k, v in inputs.items()}
    all_inputs.append(cpu_inputs)

    # move to GPU for inference
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    masks = outputs.pred_masks.cpu()
    scores = outputs.iou_scores.cpu()

    all_masks.append(masks)
    all_scores.append(scores)

torch.save(
    {
        "masks": all_masks,
        "scores": all_scores,
        "inputs": all_inputs,
        "files": png_files
    },
    "sam_outputs.pt"
)

print("done, saved sam_outputs.pt")