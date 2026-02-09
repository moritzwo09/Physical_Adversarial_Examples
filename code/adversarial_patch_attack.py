from art.attacks.evasion import AdversarialPatchPyTorch
from art.estimators.classification import PyTorchClassifier

from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_ood.utils import ToRGB
from GTSRB.gtsrb import GTSRB
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import time

trans = Compose([
            ToRGB(),
            ToTensor(),
            Resize((64, 64), antialias=True)
        ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush=True)
batch_size = 3
test_data = GTSRB(root=".", train=False, transforms=trans)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

resnet18_model = torch.load("64x64/label-net-resnet18-64.pt", map_location="cpu", weights_only=False)
wideresnet40_model = torch.load("64x64/label-net-wrn40-64.pt", map_location="cpu", weights_only=False)

resnet18_model.eval().to(device)
wideresnet40_model.eval().to(device)

digital_imgs = Path("digital_imgs_1200")
images = []
to_tensor = ToTensor()
for img in sorted(digital_imgs.glob("*.png")):
    img = Image.open(img)
    img = img.resize((64, 64))
    img_t = to_tensor(img)
    images.append(img_t)

images = torch.stack(images)
labels = [16, 1, 38, 11, 33, 18, 9, 13, 34, 2, 7, 13, 26, 15, 28, 3]
labels = np.array(labels)

sam_masks = Path("sam_masks")
mask_imgs = []
for mask_path in sorted(sam_masks.glob("*.png")):
    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize((64, 64))
    mask_np = np.array(mask_img)
    # black = traffic sign; True means allowed patch center
    mask_imgs.append(mask_np < 128)
mask_imgs = np.stack(mask_imgs)

images = images.to(device)

# vorher mischen, sonst bleibt die Reihenfolge
perm = torch.randperm(len(images))
images = images[perm]
labels = labels[perm.numpy()]
mask_imgs = mask_imgs[perm.numpy()]


model = resnet18_model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 64, 64),
    nb_classes=43,
)

#classifier.fit(train_imgs, train_labels, batch_size=batch_size, nb_epochs=3)
x_np = images.detach().cpu().numpy()
y_np = labels.astype(np.int64)

print("Start der Berechnung:", flush=True)
start_time = time.time()

attack = AdversarialPatchPyTorch(
    estimator=classifier,
    patch_shape=(3, 16, 16),
    max_iter=800,
    learning_rate=0.1,
    batch_size=4,
    scale_min=0.2,
    scale_max=0.6,
    rotation_max=45,
    targeted=False,
    verbose=False,
)

# learn one universal patch
print("Lerne Patch...", flush=True)
patch, patch_mask = attack.generate(x=x_np, y=y_np, mask=mask_imgs, verbose=True)
print(f"Patch gelernt. patch shape={patch.shape}, patch mask shape={patch_mask.shape}", flush=True)

# apply patch to all images -> 64x64 output
scale = 16 / 64
x_test_adv = attack.apply_patch(x_np, scale=scale, patch_external=patch, mask=mask_imgs)

x_test_adv = x_test_adv.astype(np.float32)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Berechnung beendet. Dauer: {elapsed_time:.2f} Sekunden ({elapsed_time/60:.2f} Minuten)", flush=True)

print(f"Shape of x_test_adv: {x_test_adv.shape}, shape of y_np: {y_np.shape}", flush=True)

predictions = classifier.predict(x_test_adv)
pred_classes_adv = np.argmax(predictions, axis=1)
accuracy_test = np.mean(pred_classes_adv[:len(y_np)] == y_np)
test_percent = float(accuracy_test) * 100.0
print(f"Accuracy on adversarial examples: {test_percent:.2f}%", flush=True)

out = {
    "x_adv": torch.from_numpy(x_test_adv),
    "x_orig": torch.from_numpy(x_np),
    "y": torch.from_numpy(y_np),
}

torch.save(out, "adversarial_patch_adv.pt")
