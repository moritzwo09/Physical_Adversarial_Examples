from art.attacks.evasion import GRAPHITEWhiteboxPyTorch
from art.estimators.classification import PyTorchClassifier

from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_ood.utils import ToRGB
from gtsrb import GTSRB
from pathlib import Path
from PIL import Image
import numpy as np

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

images = images.to(device)

# vorher mischen, sonst bleibt die Reihenfolge
perm = torch.randperm(len(images))
images = images[perm]
labels = labels[perm.numpy()]

# Splits
train_imgs, train_labels = images[:8], labels[:8]
test_imgs,  test_labels  = images[8:], labels[8:]


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
x_test_np = test_imgs.detach().cpu().numpy()
y_test_np = test_labels.astype(np.int64)

print("Test:", flush=True)
predictions = classifier.predict(x_test_np)
pred_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(pred_classes == test_labels)
print(f"Accuracy on val set: {accuracy * 100:.2f}%", flush=True)

print("Start der Berechnung:", flush=True)
attack = GRAPHITEWhiteboxPyTorch(classifier=classifier, net_size=(64, 64), num_xforms=10)
x_test_adv = attack.generate(x=x_test_np, y=y_test_np)
predictions = classifier.predict(x_test_adv)
accuracy_test = np.mean(pred_classes == test_labels)
print(f"Accuracy on test set: {accuracy_test * 100:.2f}%", flush=True)

out = {
    "x_adv": torch.from_numpy(x_test_adv),
    "x_orig": torch.from_numpy(x_test_np),
    "y": torch.from_numpy(y_test_np),
}

torch.save(out, "graphite_adv.pt")