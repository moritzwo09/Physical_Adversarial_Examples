from pathlib import Path
from PIL import Image

digital_imgs = Path("../digital_imgs")
out_dir = Path("../digital_imgs_1200")
out_dir.mkdir(exist_ok=True)

for img_path in digital_imgs.glob("*.png"):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((1200, 1200), resample=Image.BICUBIC)
    img.save(out_dir / img_path.name)