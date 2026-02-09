import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from art.attacks.evasion import AdversarialPatchPyTorch
from art.estimators.classification import PyTorchClassifier
from pytorch_ood.utils import ToRGB
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from GTSRB.gtsrb import GTSRB

# -----------------------------
# Config
# -----------------------------
TRAIN_SAMPLES = 8000
TEST_SAMPLES = 200  # small test set for printing
BATCH_SIZE = 64
PATCH_SIZE = 16
IMAGE_SIZE = 64

MODEL_PATH = "64x64/label-net-resnet18-64.pt"
OUT_FILE = "adversarial_patch_adv_split.pt"

# -----------------------------
# Setup
# -----------------------------
trans = Compose([
    ToRGB(),
    ToTensor(),
    Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush=True)

train_data = GTSRB(root=".", train=True, transforms=trans)
test_data = GTSRB(root=".", train=False, transforms=trans)


def collect_samples(dataset, num_samples, shuffle):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)
    xs = []
    ys = []
    collected = 0
    for x, y in loader:
        # y has shape [B, 3] -> class label is y[:, 0]
        y_cls = y[:, 0].long()
        xs.append(x)
        ys.append(y_cls)
        collected += x.size(0)
        if collected >= num_samples:
            break

    x_all = torch.cat(xs, dim=0)[:num_samples]
    y_all = torch.cat(ys, dim=0)[:num_samples]
    return x_all, y_all


# -----------------------------
# Load data
# -----------------------------
train_x, train_y = collect_samples(train_data, TRAIN_SAMPLES, shuffle=True)
test_x, test_y = collect_samples(test_data, TEST_SAMPLES, shuffle=False)

train_x = train_x.to(device)

# -----------------------------
# Model & ART classifier
# -----------------------------
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, IMAGE_SIZE, IMAGE_SIZE),
    nb_classes=43,
)

# -----------------------------
# Train universal patch
# -----------------------------
train_x_np = train_x.detach().cpu().numpy()
train_y_np = train_y.detach().cpu().numpy().astype(np.int64)

test_x_np = test_x.detach().cpu().numpy()
test_y_np = test_y.detach().cpu().numpy().astype(np.int64)

print("Start der Berechnung:", flush=True)
start_time = time.time()

attack = AdversarialPatchPyTorch(
    estimator=classifier,
    patch_shape=(3, PATCH_SIZE, PATCH_SIZE),
    max_iter=800,
    learning_rate=0.1,
    batch_size=16,
    scale_min=0.2,
    scale_max=0.6,
    rotation_max=45,
    targeted=True,
    verbose=True,
)

print("Lerne Patch...", flush=True)
target_y_np = np.full_like(train_y_np, 14, dtype=np.int64)
patch, patch_mask = attack.generate(x=train_x_np, y=target_y_np)
print(
    f"Patch gelernt. patch shape={patch.shape}, patch mask shape={patch_mask.shape}",
    flush=True,
)

# -----------------------------
# Apply patch on test set
# -----------------------------
scale = PATCH_SIZE / IMAGE_SIZE
x_test_adv = attack.apply_patch(test_x_np, scale=scale, patch_external=patch)

x_test_adv = x_test_adv.astype(np.float32)

end_time = time.time()
elapsed_time = end_time - start_time
print(
    f"Berechnung beendet. Dauer: {elapsed_time:.2f} Sekunden ({elapsed_time/60:.2f} Minuten)",
    flush=True,
)

predictions = classifier.predict(x_test_adv)
pred_classes_adv = np.argmax(predictions, axis=1)
accuracy_test = np.mean(pred_classes_adv[: len(test_y_np)] == test_y_np)
test_percent = float(accuracy_test) * 100.0
print(f"Accuracy on adversarial test examples: {test_percent:.2f}%", flush=True)

out = {
    "x_adv": torch.from_numpy(x_test_adv),
    "x_orig": torch.from_numpy(test_x_np),
    "y": torch.from_numpy(test_y_np),
    "patch": torch.from_numpy(patch),
    "patch_mask": torch.from_numpy(patch_mask),
}

torch.save(out, OUT_FILE)
print(f"Saved: {OUT_FILE}", flush=True)
